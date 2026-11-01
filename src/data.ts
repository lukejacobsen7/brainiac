import { App, TFile, requestUrl, getAllTags } from "obsidian";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { UMAP } from "umap-js";
import { rng } from "./shape";

export interface Entry {
	id: string; // vault path, or "claude:<sessionId>"
	kind: "note" | "chat";
	title: string;
	group: string;
	time: number; // ms
	clock: boolean; // true when the time of day is real, not a date-only guess
	parts: string[]; // one label per real paragraph (notes) or message (chats); each becomes a dot
	links: string[]; // ids this entry links to
	text: string; // embedding text only, never displayed
	resume?: string; // claude session id
}

export interface BrainSettings {
	includeChats: boolean;
	engine: "tfidf" | "ollama";
	ollamaModel: string;
	excluded: string; // comma-separated path prefixes
	palette: string;
	layout: string;
	theme: string;
	newestFirst: boolean;
	pointSize: number;
	rotate: boolean;
	rotateSpeed: number;
	links: boolean;
	bloom: boolean;
	outline: boolean;
}

export const DEFAULT_SETTINGS: BrainSettings = {
	includeChats: false, // reads ~/.claude outside the vault, so opt-in
	engine: "tfidf",
	ollamaModel: "nomic-embed-text",
	excluded: "",
	palette: "Default",
	layout: "anatomy",
	theme: "Original",
	newestFirst: false,
	pointSize: 1,
	rotate: true,
	rotateSpeed: 1,
	links: false,
	bloom: true,
	outline: true,
};

// never index or show credentials
const SECRET_NAME = /(api[\s_-]?key|token|password|passwd|secret|credential|keychain|cookie)/i;
const SECRET_TEXT = /(sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9-]+|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+|[A-Za-z0-9+/_-]{40,})/g;

export function scrub(s: string) {
	return s.replace(SECRET_TEXT, "•••");
}

// group by top-level folder; if one folder holds most of the vault, use its subfolders instead
function grouper(paths: string[]): (path: string) => string {
	const top = new Map<string, number>();
	for (const p of paths) { const i = p.indexOf("/"); if (i > 0) top.set(p.slice(0, i), (top.get(p.slice(0, i)) ?? 0) + 1); }
	let main = "";
	for (const [k, n] of top) if (n >= paths.length * 0.7) main = k;
	return path => {
		const parts = path.split("/");
		if (parts.length === 1) return "Root";
		if (parts[0] === main && parts.length > 2) return parts[1];
		return parts[0];
	};
}

function dateFrom(file: TFile, fm: Record<string, unknown> | undefined): [number, boolean] {
	for (const k of ["created", "date", "captured", "processed"]) {
		const v = fm?.[k];
		if (typeof v === "string" || v instanceof Date) {
			const t = new Date(v).getTime();
			if (!isNaN(t) && t > 0) return [t, typeof v === "string" ? /\d\d:\d\d/.test(v) : true];
		}
	}
	// "2026-10-27 06-48 - ..." session names carry the time too
	const m = file.basename.match(/(20\d\d)-?(\d\d)-?(\d\d)(?:[ T_](\d\d)[-:.](\d\d))?/);
	if (m) {
		const t = new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4] ?? "12"}:${m[5] ?? "00"}:00`).getTime();
		if (!isNaN(t)) return [t, !!m[4]];
	}
	return [file.stat.ctime, true];
}

export async function collectNotes(app: App, s: BrainSettings): Promise<Entry[]> {
	const excluded = s.excluded.split(",").map(x => x.trim()).filter(Boolean);
	const files = app.vault.getMarkdownFiles().filter(f =>
		!excluded.some(p => f.path === p || f.path.startsWith(p.endsWith("/") ? p : p + "/")) &&
		!f.path.includes("/_backups/") &&
		!SECRET_NAME.test(f.basename));
	const groupOf = grouper(files.map(f => f.path));
	const out: Entry[] = [];
	for (const f of files) {
		const cache = app.metadataCache.getFileCache(f);
		const tags = cache ? (getAllTags(cache) ?? []) : [];
		const links = Object.keys(app.metadataCache.resolvedLinks[f.path] ?? {});
		let body = "";
		try {
			body = await app.vault.cachedRead(f);
		} catch { /* unreadable, title only */ }
		body = body.replace(/^---[\s\S]*?---/, "");
		const parts = paragraphs(body);
		body = body.slice(0, 4000);
		const when = dateFrom(f, cache?.frontmatter);
		out.push({
			id: f.path,
			kind: "note",
			title: f.basename,
			group: groupOf(f.path),
			time: when[0],
			clock: when[1],
			parts,
			links,
			text: scrub(`${f.basename} ${f.basename} ${f.basename} ${tags.join(" ")} ${body}`),
		});
	}
	return out;
}

// Claude Code history: one entry per session, one dot per message the user typed.
// parsed files are cached by mtime+size for the life of the app
// the few fields of a Claude Code transcript line we read
interface ChatLine {
	type?: string;
	aiTitle?: string;
	isMeta?: boolean;
	isSidechain?: boolean;
	timestamp?: string;
	message?: { content?: string | { type?: string; text?: string }[] };
}
const chatCache = new Map<string, { mtime: number; size: number; entry: Entry | null }>();
export function collectChats(): Entry[] {
	const root = path.join(os.homedir(), ".claude", "projects");
	if (!fs.existsSync(root)) return [];
	const out: Entry[] = [];
	for (const proj of fs.readdirSync(root)) {
		if (proj.includes("private-tmp") || proj === "-") continue; // sdk scratch runs
		const dir = path.join(root, proj);
		let names: string[] = [];
		try { names = fs.readdirSync(dir).filter(n => n.endsWith(".jsonl")); } catch { continue; }
		const projName = proj.replace(/^-Users-[^-]+-?/, "").replace(/^Documents-/, "") || "home";
		for (const n of names) {
			const fp = path.join(dir, n);
			let st: fs.Stats;
			try { st = fs.statSync(fp); } catch { continue; }
			const hit = chatCache.get(fp);
			if (hit && hit.mtime === st.mtimeMs && hit.size === st.size) { if (hit.entry) out.push(hit.entry); continue; }
			let raw = "";
			try { raw = fs.readFileSync(fp, "utf8"); } catch { continue; }
			let title = "", first = "", time = 0, texts: string[] = [], users = 0;
			const msgTimes: string[] = [];
			for (const line of raw.split("\n")) {
				// cheap filters first; most lines are tool output
				const isTitle = line.includes('"ai-title"');
				if (!isTitle && (!line.includes('"type":"user"') || line.includes('"tool_result"') || line.includes('"isMeta":true') || line.includes('"isSidechain":true'))) continue;
				let j: ChatLine;
				try { j = JSON.parse(line) as ChatLine; } catch { continue; }
				if (j.type === "ai-title" && j.aiTitle) { title = j.aiTitle; continue; }
				if (j.type !== "user" || j.isMeta || j.isSidechain) continue;
				const c = j.message?.content;
				const t = typeof c === "string" ? c : Array.isArray(c) ? c.filter(b => b.type === "text").map(b => b.text ?? "").join(" ") : "";
				if (!t || t.startsWith("<")) continue;
				users++;
				msgTimes.push(j.timestamp ? `message ${users} · ${new Date(j.timestamp).toLocaleString()}` : `message ${users}`);
				if (!time && j.timestamp) time = new Date(j.timestamp).getTime();
				if (!first) first = t;
				if (texts.join(" ").length < 3000) texts.push(t);
			}
			if (users < 2 || !first) { chatCache.set(fp, { mtime: st.mtimeMs, size: st.size, entry: null }); continue; }
			const label = scrub(title || first).replace(/\s+/g, " ").slice(0, 90);
			out.push({
				id: "claude:" + n.replace(/\.jsonl$/, ""),
				kind: "chat",
				title: label,
				group: "Claude chats",
				time: time || st.mtimeMs,
				clock: true,
				parts: msgTimes.slice(0, MAX_PARTS),
				links: [],
				text: scrub(`${projName} ${title} ${texts.join(" ")}`).slice(0, 4000),
				resume: n.replace(/\.jsonl$/, ""),
			});
			chatCache.set(fp, { mtime: st.mtimeMs, size: st.size, entry: out[out.length - 1] });
		}
	}
	return out;
}

const MAX_PARTS = 2000;

// real paragraphs of a note, labelled by the heading they sit under.
// labels never include body text, so nothing sensitive reaches the tooltip
function paragraphs(body: string): string[] {
	const out: string[] = [];
	let heading = "", buf = "", inCode = false;
	const flush = () => {
		if (buf.replace(/[\s>*_\-#|`]/g, "").length >= 40) out.push(heading);
		buf = "";
	};
	for (const line of body.split("\n")) {
		if (line.trim().startsWith("```")) { inCode = !inCode; buf += line; continue; }
		const h = !inCode && line.match(/^#{1,6}\s+(.*)/);
		if (h) { flush(); heading = scrub(h[1]).slice(0, 80); continue; }
		if (!inCode && !line.trim()) flush();
		else buf += line + " ";
	}
	flush();
	return out.slice(0, MAX_PARTS);
}

const STOP = new Set("the and for that this with you are was have not but from they his her she him its our your can will just what when then than them there their about into also been were which would could should more some only other over such very each like does did done get got make made need want use used using one two all any how why who out off has had who".split(" "));

function tokens(s: string) {
	return s.toLowerCase().replace(/https?:\S+/g, " ").match(/[a-z][a-z0-9]{2,24}/g)?.filter(t => !STOP.has(t)) ?? [];
}

function hashStr(s: string) {
	let h = 2166136261;
	for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
	return h >>> 0;
}

// TF-IDF with feature hashing into a dense vector, L2-normalized
export function tfidfVectors(entries: Entry[], dim = 256): number[][] {
	const docs = entries.map(e => tokens(e.text));
	const df = new Map<string, number>();
	for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
	const N = docs.length;
	return docs.map((d, i) => {
		const v = new Array<number>(dim).fill(0);
		const tf = new Map<string, number>();
		for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
		for (const [t, c] of tf) {
			const n = df.get(t)!;
			if (n < 2 || n > N * 0.4) continue;
			const w = (1 + Math.log(c)) * Math.log(N / n);
			const h = hashStr(t);
			v[h % dim] += (h & 0x80000000 ? -1 : 1) * w;
		}
		// links pull linked notes together
		for (const l of entries[i].links) { const h = hashStr("link:" + l); v[h % dim] += 1.5; }
		const norm = Math.hypot(...v) || 1;
		return v.map(x => x / norm);
	});
}

export async function ollamaVectors(entries: Entry[], model: string, onProgress: (f: number) => void): Promise<number[][]> {
	const out: number[][] = [];
	for (let i = 0; i < entries.length; i += 32) {
		const batch = entries.slice(i, i + 32).map(e => "clustering: " + e.text.slice(0, 2000));
		const r = await requestUrl({
			url: "http://localhost:11434/api/embed",
			method: "POST",
			contentType: "application/json",
			body: JSON.stringify({ model, input: batch }),
		});
		out.push(...(r.json as { embeddings: number[][] }).embeddings);
		onProgress(Math.min(1, (i + 32) / entries.length));
	}
	return out;
}

// UMAP to 3D, then rank-normalize each axis within each group so every
// group fills its own region of the brain
export async function layout(vectors: number[][], entries: Entry[], onProgress: (f: number) => void): Promise<{ u: number[][]; g: number[][] }> {
	const umap = new UMAP({ nComponents: 3, nNeighbors: Math.min(15, vectors.length - 1), minDist: 0.15, random: rng(42) });
	const nEpochs = umap.initializeFit(vectors);
	for (let i = 0; i < nEpochs; i++) {
		umap.step();
		if (i % 10 === 0) { onProgress(i / nEpochs); await new Promise(r => window.setTimeout(r, 0)); }
	}
	const emb = umap.getEmbedding();
	const u: number[][] = emb.map(() => [0, 0, 0]);
	const byGroup = new Map<string, number[]>();
	entries.forEach((e, i) => { if (!byGroup.has(e.group)) byGroup.set(e.group, []); byGroup.get(e.group)!.push(i); });
	for (const idx of byGroup.values()) {
		for (let axis = 0; axis < 3; axis++) {
			const sorted = [...idx].sort((a, b) => emb[a][axis] - emb[b][axis]);
			sorted.forEach((k, rank) => { u[k][axis] = (rank + 0.5) / sorted.length; });
		}
	}
	// galaxy: global coords centred and scaled to a ~2.2 radius
	const c = [0, 1, 2].map(a => emb.reduce((n, p) => n + p[a], 0) / emb.length);
	const d = emb.map(p => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])).sort((a, b) => a - b);
	const scale = 2.2 / (d[Math.floor(d.length * 0.97)] || 1);
	const g = emb.map(p => [(p[0] - c[0]) * scale, (p[1] - c[1]) * scale, (p[2] - c[2]) * scale]);
	return { u, g };
}
