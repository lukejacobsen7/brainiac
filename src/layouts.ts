// Every non-anatomy form. Each one places an anchor per entry (note or chat);
// the view scatters that entry's paragraph/message dots around its anchor.
import type { Entry } from "./data";
import { Vec3, sphereCenter, spherePoint, rng } from "./shape";

export interface Ctx {
	entries: Entry[];
	gi: number[]; // group index per entry
	groups: { name: string; count: number; dots: number }[];
	u: Vec3[]; // per-group semantic coords in [0,1]^3
	g: Vec3[]; // global semantic coords (galaxy)
	tMin: number;
	tMax: number;
}

export interface Label { text: string; p: Vec3; kind: "group" | "axis" }

export interface LayoutResult {
	anchor: Float32Array; // entries * 3
	spread: Float32Array; // per entry: how far its dots scatter
	flat: boolean; // 2D chart: face the camera, no spinning
	labels: Label[];
	stat?: string;
}

export interface FormDef { id: string; name: string; hint: string }

export const FORMS: FormDef[] = [
	{ id: "anatomy", name: "Anatomy", hint: "brain, spine and nerves" },
	{ id: "galaxy", name: "Galaxy", hint: "free cloud, similar notes close" },
	{ id: "spheres", name: "Spheres", hint: "one ball per folder" },
	{ id: "clusters", name: "Clusters", hint: "balls by topic, not folder" },
	{ id: "shells", name: "Shells", hint: "one shell per folder" },
	{ id: "wheel", name: "Wheel", hint: "folder slices, older inside" },
	{ id: "ribbon", name: "Ribbon", hint: "folders × months" },
	{ id: "lifelines", name: "Lifelines", hint: "each folder over time" },
	{ id: "stream", name: "Stream", hint: "how much you wrote, stacked" },
	{ id: "folders", name: "Folders", hint: "boxes sized by folder" },
	{ id: "calendar", name: "Calendar", hint: "every day you wrote" },
	{ id: "clock", name: "Clock", hint: "what hour you write" },
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const GOLDEN = Math.PI * (3 - Math.sqrt(5));

function result(n: number, flat: boolean): LayoutResult {
	return { anchor: new Float32Array(n * 3), spread: new Float32Array(n), flat, labels: [] };
}

function tNorm(ctx: Ctx, t: number) {
	return Math.min(1, Math.max(0, (t - ctx.tMin) / Math.max(1, ctx.tMax - ctx.tMin)));
}

function monthKey(t: number) { const d = new Date(t); return d.getFullYear() * 12 + d.getMonth(); }

// pack items into a cell as a sunflower so cells read as dense blobs
function packCells(keys: (string | number)[], radiusOf: (k: string | number, n: number) => number) {
	const lists = new Map<string | number, number[]>();
	keys.forEach((k, i) => { if (!lists.has(k)) lists.set(k, []); lists.get(k)!.push(i); });
	const off: [number, number][] = keys.map(() => [0, 0]);
	for (const [k, idx] of lists) {
		const R = radiusOf(k, idx.length);
		idx.forEach((i, j) => {
			const r = R * Math.sqrt((j + 0.5) / idx.length);
			off[i] = [Math.cos(j * GOLDEN) * r, Math.sin(j * GOLDEN) * r];
		});
	}
	return off;
}

function groupLabels(ctx: Ctx, res: LayoutResult) {
	const sum = ctx.groups.map(() => [0, 0, 0, 0]);
	ctx.entries.forEach((_, i) => {
		const s = sum[ctx.gi[i]];
		s[0] += res.anchor[i * 3]; s[1] += res.anchor[i * 3 + 1]; s[2] += res.anchor[i * 3 + 2]; s[3]++;
	});
	ctx.groups.forEach((g, i) => {
		const s = sum[i];
		if (s[3]) res.labels.push({ text: g.name, p: [s[0] / s[3], s[1] / s[3], s[2] / s[3]], kind: "group" });
	});
}

function galaxy(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, false);
	ctx.g.forEach((p, i) => { res.anchor.set(p, i * 3); res.spread[i] = 0.05; });
	groupLabels(ctx, res);
	return res;
}

function spheres(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, false);
	const r = ctx.groups.map(g => 0.07 * Math.cbrt(g.count) + 0.05);
	const c = ctx.groups.map((_, i) => sphereCenter(i, ctx.groups.length));
	const rnd = rng(11);
	ctx.entries.forEach((_, i) => {
		const k = ctx.gi[i];
		res.anchor.set(spherePoint(c[k], r[k], ctx.u[i], rnd), i * 3);
		res.spread[i] = 0.05 * r[k] + 0.01;
	});
	groupLabels(ctx, res);
	return res;
}

// k-means on galaxy coords; balls named by the most telling title word
function clusters(ctx: Ctx): LayoutResult {
	const n = ctx.entries.length;
	const k = Math.min(14, Math.max(2, Math.round(Math.sqrt(n / 12))));
	const rnd = rng(5);
	let cent: Vec3[] = Array.from({ length: k }, () => ctx.g[Math.floor(rnd() * n)]);
	const lab = new Int32Array(n);
	for (let it = 0; it < 25; it++) {
		for (let i = 0; i < n; i++) {
			let best = 0, bd = Infinity;
			for (let c = 0; c < k; c++) {
				const d = (ctx.g[i][0] - cent[c][0]) ** 2 + (ctx.g[i][1] - cent[c][1]) ** 2 + (ctx.g[i][2] - cent[c][2]) ** 2;
				if (d < bd) { bd = d; best = c; }
			}
			lab[i] = best;
		}
		const acc = Array.from({ length: k }, () => [0, 0, 0, 0]);
		for (let i = 0; i < n; i++) { const a = acc[lab[i]]; a[0] += ctx.g[i][0]; a[1] += ctx.g[i][1]; a[2] += ctx.g[i][2]; a[3]++; }
		cent = acc.map((a, c) => a[3] ? [a[0] / a[3], a[1] / a[3], a[2] / a[3]] as Vec3 : cent[c]);
	}
	const sizes = Array.from({ length: k }, (_, c) => ({ c, n: 0 }));
	for (let i = 0; i < n; i++) sizes[lab[i]].n++;
	const order = sizes.filter(s => s.n).sort((a, b) => b.n - a.n);
	const slot = new Map(order.map((s, i) => [s.c, i]));
	// per-cluster rank-normalised coords so each ball is filled
	const local: Vec3[] = ctx.g.map(() => [0.5, 0.5, 0.5]);
	for (const s of order) {
		const idx: number[] = [];
		for (let i = 0; i < n; i++) if (lab[i] === s.c) idx.push(i);
		for (let a = 0; a < 3; a++) [...idx].sort((x, y) => ctx.g[x][a] - ctx.g[y][a]).forEach((i, r) => { local[i][a] = (r + 0.5) / idx.length; });
	}
	// word that best separates each cluster's titles from the rest
	const words = ctx.entries.map(e => new Set((e.title.toLowerCase().match(/[a-z][a-z0-9]{3,}/g) ?? []).filter(w => !STOP.has(w))));
	const total = new Map<string, number>();
	words.forEach(ws => ws.forEach(w => total.set(w, (total.get(w) ?? 0) + 1)));
	const res = result(n, false);
	const rr = rng(12);
	for (const s of order) {
		const cnt = new Map<string, number>();
		for (let i = 0; i < n; i++) if (lab[i] === s.c) words[i].forEach(w => cnt.set(w, (cnt.get(w) ?? 0) + 1));
		let best = "", score = 0;
		for (const [w, c] of cnt) { if (c < 2) continue; const sc = c * Math.log(n / total.get(w)!); if (sc > score) { score = sc; best = w; } }
		const i0 = slot.get(s.c)!;
		const center = sphereCenter(i0, order.length);
		const rad = 0.07 * Math.cbrt(s.n) + 0.05;
		res.labels.push({ text: `${best || "misc"} · ${s.n}`, p: center, kind: "group" });
		for (let i = 0; i < n; i++) if (lab[i] === s.c) {
			res.anchor.set(spherePoint(center, rad, local[i], rr), i * 3);
			res.spread[i] = 0.05 * rad + 0.01;
		}
	}
	return res;
}

function shells(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, false);
	const G = ctx.groups.length;
	const rad = ctx.groups.map((_, i) => 0.45 + 2.0 * Math.sqrt((i + 0.5) / G));
	ctx.entries.forEach((_, i) => {
		const k = ctx.gi[i], u = ctx.u[i];
		const th = u[0] * Math.PI * 2, ph = Math.acos(2 * u[1] - 1);
		const r = rad[k] * (0.97 + u[2] * 0.06);
		res.anchor.set([Math.sin(ph) * Math.cos(th) * r, Math.cos(ph) * r, Math.sin(ph) * Math.sin(th) * r], i * 3);
		res.spread[i] = 0.025;
	});
	// spread labels around so shells don't stack them
	ctx.groups.forEach((g, i) => { const a = Math.PI / 2 + i * GOLDEN; res.labels.push({ text: g.name, p: [Math.cos(a) * rad[i], Math.sin(a) * rad[i], 0], kind: "group" }); });
	return res;
}

function wheel(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const total = ctx.groups.reduce((n, g) => n + Math.max(g.count, ctx.entries.length * 0.012), 0);
	let at = 0;
	const sector = ctx.groups.map(g => { const w = Math.max(g.count, ctx.entries.length * 0.012) / total * Math.PI * 2; const s = [at, w]; at += w; return s; });
	ctx.entries.forEach((e, i) => {
		const [a0, w] = sector[ctx.gi[i]];
		const ang = a0 + w * (0.05 + 0.9 * ctx.u[i][0]);
		const r = 0.35 + 2.2 * Math.sqrt(tNorm(ctx, e.time));
		res.anchor.set([Math.cos(ang) * r, Math.sin(ang) * r, (ctx.u[i][1] - 0.5) * 0.15], i * 3);
		res.spread[i] = 0.03;
	});
	ctx.groups.forEach((g, i) => {
		if (sector[i][1] < 0.14) return; // too thin to label
		const mid = sector[i][0] + sector[i][1] / 2;
		res.labels.push({ text: g.name, p: [Math.cos(mid) * 2.75, Math.sin(mid) * 2.75, 0], kind: "group" });
	});
	return res;
}

// rows = folders, columns = months
function ribbon(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const m0 = monthKey(ctx.tMin), m1 = monthKey(ctx.tMax);
	const M = Math.max(1, m1 - m0 + 1), G = ctx.groups.length;
	const W = 6, dx = W / M, dy = Math.min(0.32, 4 / G);
	const x0 = -W / 2, y0 = (G - 1) * dy / 2;
	const keys = ctx.entries.map((e, i) => `${ctx.gi[i]}:${Math.min(M - 1, Math.max(0, monthKey(e.time) - m0))}`);
	const cell = Math.min(dx, dy) * 0.46;
	const maxN = Math.max(...[...new Set(keys)].map(k => keys.filter(x => x === k).length));
	const off = packCells(keys, (_, n) => cell * Math.sqrt(n / maxN) + cell * 0.15);
	ctx.entries.forEach((e, i) => {
		const col = Math.min(M - 1, Math.max(0, monthKey(e.time) - m0));
		res.anchor.set([x0 + (col + 0.5) * dx + off[i][0], y0 - ctx.gi[i] * dy + off[i][1], 0], i * 3);
		res.spread[i] = cell * 0.25;
	});
	ctx.groups.forEach((g, i) => res.labels.push({ text: g.name, p: [x0 - 0.15, y0 - i * dy, 0], kind: "axis" }));
	const every = Math.max(1, Math.ceil(M / 12));
	for (let c = 0; c < M; c += every) {
		const mk = m0 + c;
		res.labels.push({ text: `${MONTHS[mk % 12]} ${String(Math.floor(mk / 12)).slice(2)}`, p: [x0 + (c + 0.5) * dx, y0 - G * dy - 0.05, 0], kind: "axis" });
	}
	return res;
}

function lifelines(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const G = ctx.groups.length, W = 6, dy = Math.min(0.32, 4 / G);
	const x0 = -W / 2, y0 = (G - 1) * dy / 2;
	const first = ctx.groups.map(() => Infinity), last = ctx.groups.map(() => -Infinity);
	ctx.entries.forEach((e, i) => {
		const k = ctx.gi[i];
		first[k] = Math.min(first[k], e.time); last[k] = Math.max(last[k], e.time);
		res.anchor.set([x0 + tNorm(ctx, e.time) * W, y0 - k * dy + (ctx.u[i][1] - 0.5) * dy * 0.35, 0], i * 3);
		res.spread[i] = dy * 0.12;
	});
	const fmt = (t: number) => new Date(t).toLocaleDateString(undefined, { month: "short", year: "2-digit" });
	ctx.groups.forEach((g, i) => res.labels.push({ text: `${g.name}  ${fmt(first[i])}–${fmt(last[i])}`, p: [x0 - 0.15, y0 - i * dy, 0], kind: "axis" }));
	return res;
}

// stacked, centred bands of dots per folder per month
function stream(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const m0 = monthKey(ctx.tMin), m1 = monthKey(ctx.tMax);
	const M = Math.max(1, m1 - m0 + 1), G = ctx.groups.length, W = 6, dx = W / M;
	const vol = Array.from({ length: M }, () => new Array<number>(G).fill(0));
	const col = ctx.entries.map(e => Math.min(M - 1, Math.max(0, monthKey(e.time) - m0)));
	ctx.entries.forEach((e, i) => { vol[col[i]][ctx.gi[i]] += e.parts.length + 1; });
	const maxT = Math.max(...vol.map(v => v.reduce((a, b) => a + b, 0))) || 1;
	const H = 3.2 / maxT;
	const band = vol.map(v => {
		let y = -v.reduce((a, b) => a + b, 0) * H / 2;
		return v.map(c => { const b = [y, y + c * H]; y += c * H; return b; });
	});
	ctx.entries.forEach((e, i) => {
		const [b0, b1] = band[col[i]][ctx.gi[i]];
		res.anchor.set([-W / 2 + (col[i] + ctx.u[i][0]) * dx, b0 + (b1 - b0) * ctx.u[i][1], 0], i * 3);
		res.spread[i] = Math.min(dx * 0.5, (b1 - b0) * 0.5);
	});
	// label each folder where its band is thickest
	ctx.groups.forEach((g, k) => {
		let best = 0, bm = 0;
		for (let m = 0; m < M; m++) if (vol[m][k] > best) { best = vol[m][k]; bm = m; }
		if (best * H > 0.08) res.labels.push({ text: g.name, p: [-W / 2 + (bm + 0.5) * dx, (band[bm][k][0] + band[bm][k][1]) / 2, 0], kind: "group" });
	});
	const every = Math.max(1, Math.ceil(M / 12));
	for (let c = 0; c < M; c += every) {
		const mk = m0 + c;
		res.labels.push({ text: `${MONTHS[mk % 12]} ${String(Math.floor(mk / 12)).slice(2)}`, p: [-W / 2 + (c + 0.5) * dx, -1.85, 0], kind: "axis" });
	}
	return res;
}

// squarified treemap of folders by dot count
function folders(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const items = ctx.groups.map((g, i) => ({ i, w: g.dots }));
	const rects = new Array<[number, number, number, number]>(items.length);
	const total = items.reduce((n, it) => n + it.w, 0);
	const W = 5.8, H = 3.6;
	let x = -W / 2, y = -H / 2, w = W, h = H;
	let row: typeof items = [];
	const scale = (W * H) / total;
	const worst = (r: typeof items, side: number) => {
		const s = r.reduce((n, it) => n + it.w * scale, 0);
		const mx = Math.max(...r.map(it => it.w * scale)), mn = Math.min(...r.map(it => it.w * scale));
		return Math.max(side * side * mx / (s * s), (s * s) / (side * side * mn));
	};
	const layoutRow = (r: typeof items) => {
		const s = r.reduce((n, it) => n + it.w * scale, 0);
		if (w >= h) { const cw = s / h; let yy = y; for (const it of r) { const ch = it.w * scale / cw; rects[it.i] = [x, yy, cw, ch]; yy += ch; } x += cw; w -= cw; }
		else { const ch = s / w; let xx = x; for (const it of r) { const cw = it.w * scale / ch; rects[it.i] = [xx, y, cw, ch]; xx += cw; } y += ch; h -= ch; }
	};
	for (const it of items) {
		const side = Math.min(w, h);
		if (!row.length || worst([...row, it], side) <= worst(row, side)) row.push(it);
		else { layoutRow(row); row = [it]; }
	}
	if (row.length) layoutRow(row);
	ctx.entries.forEach((_, i) => {
		const [rx, ry, rw, rh] = rects[ctx.gi[i]];
		const pad = Math.min(rw, rh) * 0.06;
		res.anchor.set([rx + pad + ctx.u[i][0] * (rw - 2 * pad), -(ry + pad + ctx.u[i][1] * (rh - 2 * pad)), 0], i * 3);
		res.spread[i] = Math.min(rw, rh) * 0.025;
	});
	ctx.groups.forEach((g, i) => { const [rx, ry, rw] = rects[i]; if (rw > 0.25) res.labels.push({ text: g.name, p: [rx + 0.08, -ry - 0.06, 0], kind: "axis" }); });
	return res;
}

// GitHub-style: a row of weeks per year, a cell per day
function calendar(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const day = (t: number) => { const d = new Date(t); d.setHours(0, 0, 0, 0); return d; };
	const y0 = new Date(ctx.tMin).getFullYear(), y1 = new Date(ctx.tMax).getFullYear();
	const Y = y1 - y0 + 1, c = 0.06, blockH = 8 * c + 0.12;
	const top = (Y - 1) * blockH / 2 + 3.5 * c;
	const cellOf = (t: number) => {
		const d = day(t), jan1 = new Date(d.getFullYear(), 0, 1);
		const doy = Math.round((d.getTime() - jan1.getTime()) / 864e5);
		const week = Math.floor((doy + jan1.getDay()) / 7);
		return { x: -53 * c / 2 + (week + 0.5) * c, y: top - (d.getFullYear() - y0) * blockH - (d.getDay() + 0.5) * c };
	};
	const keys = ctx.entries.map(e => day(e.time).getTime());
	const off = packCells(keys, () => c * 0.42);
	ctx.entries.forEach((e, i) => {
		const p = cellOf(e.time);
		res.anchor.set([p.x + off[i][0], p.y + off[i][1], 0], i * 3);
		res.spread[i] = c * 0.18;
	});
	for (let yy = y0; yy <= y1; yy++) res.labels.push({ text: String(yy), p: [-53 * c / 2 - 0.12, top - (yy - y0) * blockH - 3.5 * c, 0], kind: "axis" });
	for (let m = 0; m < 12; m++) { const p = cellOf(new Date(y0, m, 1).getTime()); res.labels.push({ text: MONTHS[m], p: [p.x, top + 0.1, 0], kind: "axis" }); }
	// days written + longest run
	const days = [...new Set(keys)].sort((a, b) => a - b);
	let run = 1, best = 1, bestEnd = days[0];
	for (let i = 1; i < days.length; i++) {
		run = Math.round((days[i] - days[i - 1]) / 864e5) === 1 ? run + 1 : 1;
		if (run > best) { best = run; bestEnd = days[i]; }
	}
	res.stat = `Written on ${days.length.toLocaleString()} days. Longest run: ${best} days in a row, to ${new Date(bestEnd).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" })}.`;
	return res;
}

// angle = hour of day, distance = how recent
function clock(ctx: Ctx): LayoutResult {
	const res = result(ctx.entries.length, true);
	const hours = new Array<number>(24).fill(0);
	ctx.entries.forEach((e, i) => {
		if (!e.clock) {
			// no real time of day: park in the hub so it doesn't fake a noon spike
			const a = ctx.u[i][0] * Math.PI * 2, r = 0.22 * Math.sqrt(ctx.u[i][1]);
			res.anchor.set([Math.cos(a) * r, Math.sin(a) * r, 0], i * 3);
			res.spread[i] = 0.02;
			return;
		}
		const d = new Date(e.time);
		const h = d.getHours() + d.getMinutes() / 60;
		hours[d.getHours()] += e.parts.length + 1;
		const ang = Math.PI / 2 - (h / 24) * Math.PI * 2;
		const r = 0.45 + 2.0 * tNorm(ctx, e.time);
		res.anchor.set([Math.cos(ang) * r, Math.sin(ang) * r, (ctx.u[i][2] - 0.5) * 0.1], i * 3);
		res.spread[i] = 0.035;
	});
	for (let h = 0; h < 24; h += 3) {
		const ang = Math.PI / 2 - (h / 24) * Math.PI * 2;
		res.labels.push({ text: `${String(h).padStart(2, "0")}:00`, p: [Math.cos(ang) * 2.7, Math.sin(ang) * 2.7, 0], kind: "axis" });
	}
	res.labels.push({ text: "no time", p: [0, -0.32, 0], kind: "axis" });
	const peak = hours.indexOf(Math.max(...hours));
	res.stat = `You write most around ${String(peak).padStart(2, "0")}:00. Older entries sit near the middle.`;
	return res;
}

const STOP = new Set("with that this from your have what when then than them there their about into also been were which would could should more some only other over such very each like does done make made need want used using claude session sessions video note notes".split(" "));

export const LAYOUTS: Record<string, (ctx: Ctx) => LayoutResult> = {
	galaxy, spheres, clusters, shells, wheel, ribbon, lifelines, stream, folders, calendar, clock,
};
