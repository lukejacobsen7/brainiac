import { ItemView, Keymap, Notice, WorkspaceLeaf, TFile } from "obsidian";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type BrainiacPlugin from "./main";
import { FORMS, LAYOUTS, Ctx, Label } from "./layouts";
import { Entry, collectNotes, collectChats, tfidfVectors, ollamaVectors, layout } from "./data";
import { regionByName, outlinePoints, rng, Vec3, Region, cortexPool, Snapper, partitionPool } from "./shape";

export const VIEW_TYPE = "brainiac-view";

// background colour themes, as in mindcorpus
const THEMES: Record<string, number> = { Original: 0x000000, Pine: 0x0a1810, Plum: 0x170c1c, Embers: 0x1a0b08 };

const PALETTES: Record<string, string[]> = {
	Default: [],
	Lava: ["#3a0606", "#9e1b0f", "#e2401c", "#ff8a3d", "#ffd27a"],
	Ocean: ["#06264a", "#0f5c99", "#1fa3c9", "#6fe0e6", "#d8fbff"],
	Forest: ["#0b2d14", "#1d6b33", "#4fae4a", "#a6dc63", "#eaf7b0"],
	Aurora: ["#3b1c7a", "#6c3fd1", "#2fb5c9", "#4fe39a", "#e8ff8a"],
	Mono: ["#5a5f6b", "#8a909c", "#b9bec8", "#e2e5ea", "#ffffff"],
	Sunset: ["#3d0f4f", "#9c2a6b", "#e5585a", "#f99c4a", "#ffe08a"],
};

// big folders get the big cortical lobes; the rest share the nerve tree
const EXTRA_REGIONS = ["cerebellum", "brainstem"];

const VERT = `
uniform float uMix, uTime, uSize, uPix, uFocus;
attribute vec3 posA, posB, color;
attribute float birth, size, sel;
varying vec3 vC; varying float vA;
void main() {
	vec3 p = mix(posA, posB, uMix);
	float age = uTime - birth;
	vA = age < 0.0 ? 0.0 : smoothstep(0.0, 0.015, age);
	vA *= mix(1.0, 0.08 + 0.92 * sel, uFocus);
	vC = color * (1.0 + 2.0 * exp(-age * 80.0));
	vec4 mv = modelViewMatrix * vec4(p, 1.0);
	gl_Position = projectionMatrix * mv;
	gl_PointSize = vA <= 0.0 ? 0.0 : uSize * size * uPix * (1.0 + sel * uFocus * 0.5) / -mv.z;
}`;
const FRAG = `
varying vec3 vC; varying float vA;
void main() {
	float d = length(gl_PointCoord - 0.5);
	float a = exp(-d * d * 18.0) * vA;
	if (a < 0.004) discard;
	gl_FragColor = vec4(vC * a, a);
}`;
const DUST_VERT = `
uniform float uMix, uPix, uSize;
void main() {
	vec4 mv = modelViewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * mv;
	gl_PointSize = (1.0 - uMix) * 6.0 * uSize * uPix / -mv.z;
}`;
const DUST_FRAG = `
uniform float uMix; uniform vec3 uColor;
void main() {
	float d = length(gl_PointCoord - 0.5);
	float a = exp(-d * d * 18.0) * 0.22 * (1.0 - uMix);
	if (a < 0.003) discard;
	gl_FragColor = vec4(uColor * a, a);
}`;
const LINE_VERT = `
uniform float uMix, uTime;
attribute vec3 posA, posB; attribute float birth;
varying float vA;
void main() {
	vA = uTime >= birth ? 1.0 : 0.0;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(posA, posB, uMix), 1.0);
}`;
const LINE_FRAG = `
varying float vA; uniform vec3 uColor;
void main() { if (vA < 0.5) discard; gl_FragColor = vec4(uColor * 0.07, 0.07); }`;

interface Group { name: string; count: number; dots: number; color: THREE.Color }

export class BrainView extends ItemView {
	plugin: BrainiacPlugin;
	private renderer?: THREE.WebGLRenderer;
	private composer?: EffectComposer;
	private bloom?: UnrealBloomPass;
	private scene = new THREE.Scene();
	private camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
	private controls?: OrbitControls;
	private world = new THREE.Group();
	private points?: THREE.Points;
	private dust?: THREE.Points;
	private lines?: THREE.LineSegments;
	private pointMat?: THREE.ShaderMaterial;
	private dustMat?: THREE.ShaderMaterial;
	private lineMat?: THREE.ShaderMaterial;
	private raf = 0;
	private ro?: ResizeObserver;

	private entries: Entry[] = [];
	private groups: Group[] = [];
	private primaryIdx: number[] = []; // point index of each entry's primary particle
	private ownerOf: Int32Array = new Int32Array(0); // point -> entry
	private partDim = 1;
	private partOf: Int16Array = new Int16Array(0); // point -> paragraph/message index, -1 = the note itself
	private births: number[] = [];
	private linkEnds: number[] = []; // point index for each line vertex
	private tMin = 0;
	private tMax = 1;

	private mix = 0;
	private targetMix = 0;
	private time = 1;
	private playing = false;
	private dirty = 0;
	private down?: { x: number; y: number };
	private hover = -1;
	private targetCenter = new THREE.Vector3(0, -1, 0);
	private zoomTo: number | null = null;

	// dom
	private host!: HTMLElement;
	private statusEl!: HTMLElement;
	private tip!: HTMLElement;
	private labelLayer!: HTMLElement;
	private legend!: HTMLElement;
	private slider!: HTMLInputElement;
	private dateEl!: HTMLElement;
	private playBtn!: HTMLElement;
	private spinBtn!: HTMLElement;
	private spinCheck?: HTMLInputElement;
	private searchEl!: HTMLInputElement;
	private focusGroup: string | null = null;
	private focusMonth: number | null = null;
	private form = "anatomy";
	private anatomyPos?: Float32Array;
	private pointRand?: Float32Array;
	private uE: Vec3[] = [];
	private gE: Vec3[] = [];
	private gIdx: number[] = [];
	private rawBirth: number[] = [];
	private formLabels: { l: Label; el: HTMLElement }[] = [];
	private flat = false;
	private aimFront = false;
	private dustAmt = 1;
	private trail?: THREE.Line;
	private statEl!: HTMLElement;
	private hubPanel!: HTMLElement;
	private tlPanel!: HTMLElement;
	private formBtns = new Map<string, HTMLElement>();
	private countText = "";

	constructor(leaf: WorkspaceLeaf, plugin: BrainiacPlugin) {
		super(leaf);
		this.plugin = plugin;
	}
	getViewType() { return VIEW_TYPE; }
	getDisplayText() { return "Brainiac"; }
	getIcon() { return "brain"; }

	async onOpen() {
		const root = this.contentEl;
		root.empty();
		root.addClass("mb-root");
		this.host = root.createDiv("mb-canvas");
		this.labelLayer = root.createDiv("mb-labels");
		this.tip = root.createDiv("mb-tip");

		const top = root.createDiv("mb-top");
		this.searchEl = top.createEl("input", { type: "search", placeholder: "Search notes…", cls: "mb-search" });
		this.searchEl.addEventListener("input", () => { this.focusGroup = null; this.focusMonth = null; this.drawTrail(); this.renderHub(); this.applyFocus(); });
		this.statusEl = top.createDiv("mb-status");
		this.statEl = top.createDiv("mb-stat");
		const tlBtn = top.createEl("button", { text: "When you wrote", cls: "mb-tl-btn" });
		tlBtn.onclick = () => { this.tlPanel.toggleClass("is-open", !this.tlPanel.hasClass("is-open")); this.renderTimeline(); };
		this.tlPanel = root.createDiv("mb-side mb-side-left");
		this.hubPanel = root.createDiv("mb-side mb-side-right");

		this.buildPanel(root);

		const bottom = root.createDiv("mb-bottom");
		const tl = bottom.createDiv("mb-timeline");
		this.playBtn = tl.createEl("button", { text: "▶", cls: "mb-play" });
		this.playBtn.onclick = () => this.togglePlay();
		this.spinBtn = tl.createEl("button", { cls: "mb-play" });
		this.spinBtn.onclick = () => { const st = this.plugin.settings; st.rotate = !st.rotate; void this.plugin.saveSettings(); this.syncSpin(); };
		this.syncSpin();
		this.slider = tl.createEl("input", { type: "range", cls: "mb-slider" });
		this.slider.min = "0"; this.slider.max = "1000"; this.slider.value = "1000";
		this.slider.oninput = () => { this.playing = false; this.playBtn.setText("▶"); this.setTime(+this.slider.value / 1000); };
		this.dateEl = tl.createDiv("mb-date");
		this.legend = bottom.createDiv("mb-legend");

		this.initThree();
		this.applyTheme();
		this.registerEvent(this.app.vault.on("create", () => this.markDirty()));
		this.registerEvent(this.app.vault.on("delete", () => this.markDirty()));
		this.registerEvent(this.app.vault.on("rename", () => this.markDirty()));

		this.app.workspace.onLayoutReady(() => void this.build(false));
	}

	async onClose() {
		window.cancelAnimationFrame(this.raf);
		this.ro?.disconnect();
		this.controls?.dispose();
		this.disposeScene();
		this.composer?.dispose();
		this.renderer?.dispose();
		this.renderer?.forceContextLoss();
		this.renderer?.domElement.remove();
	}

	// ---------- ui ----------
	private buildPanel(root: HTMLElement) {
		const s = this.plugin.settings;
		const toggle = root.createEl("button", { text: "☰", cls: "mb-panel-toggle" });
		const panel = root.createDiv("mb-panel");
		toggle.onclick = () => panel.toggleClass("is-open", !panel.hasClass("is-open"));

		const section = (t: string) => panel.createDiv({ text: t, cls: "mb-sec" });
		const row = (label: string) => { const r = panel.createDiv("mb-row"); r.createSpan({ text: label }); return r; };
		const check = (label: string, get: () => boolean, set: (v: boolean) => void) => {
			const r = row(label);
			const c = r.createEl("input", { type: "checkbox" });
			c.checked = get();
			c.onchange = () => { set(c.checked); void this.plugin.saveSettings(); };
		};
		const range = (label: string, min: number, max: number, step: number, get: () => number, set: (v: number) => void) => {
			const r = row(label);
			const c = r.createEl("input", { type: "range" });
			c.min = String(min); c.max = String(max); c.step = String(step); c.value = String(get());
			c.oninput = () => { set(+c.value); void this.plugin.saveSettings(); };
		};

		section("FORM");
		const grid = panel.createDiv("mb-forms");
		for (const f of FORMS) {
			const btn = grid.createEl("button", { text: f.name, cls: "mb-seg" });
			btn.setAttr("title", f.hint);
			if (s.layout === f.id) btn.addClass("is-on");
			btn.onclick = () => { s.layout = f.id; void this.plugin.saveSettings(); this.applyForm(f.id); };
			this.formBtns.set(f.id, btn);
		}
		const ord = row("Order").createEl("select");
		ord.createEl("option", { text: "Oldest first", value: "old" });
		ord.createEl("option", { text: "Newest first", value: "new" });
		ord.value = s.newestFirst ? "new" : "old";
		ord.onchange = () => { s.newestFirst = ord.value === "new"; void this.plugin.saveSettings(); this.applyBirths(); };

		section("APPEARANCE");
		const th = row("Colour theme").createEl("select");
		for (const k of Object.keys(THEMES)) th.createEl("option", { text: k, value: k });
		th.value = s.theme;
		th.onchange = () => { s.theme = th.value; void this.plugin.saveSettings(); this.applyTheme(); };
		const pal = row("Data colours").createEl("select");
		for (const k of Object.keys(PALETTES)) pal.createEl("option", { text: k, value: k });
		pal.value = s.palette;
		pal.onchange = () => { s.palette = pal.value; void this.plugin.saveSettings(); this.recolor(); };
		range("Point size", 0.3, 3, 0.05, () => s.pointSize, v => { s.pointSize = v; if (this.pointMat) this.pointMat.uniforms.uSize.value = v * 22; if (this.dustMat) this.dustMat.uniforms.uSize.value = v; });
		check("Links", () => s.links, v => { s.links = v; if (this.lines) this.lines.visible = v; });
		check("Glow", () => s.bloom, v => { s.bloom = v; });
		check("Brain outline", () => s.outline, v => { s.outline = v; if (this.dust) this.dust.visible = v; });

		section("MOTION");
		check("Turning", () => s.rotate, v => { s.rotate = v; this.syncSpin(); });
		this.spinCheck = panel.findAll("input[type=checkbox]").pop() as HTMLInputElement;
		range("Speed", 0.1, 4, 0.1, () => s.rotateSpeed, v => { s.rotateSpeed = v; });

		section("DATA");
		check("Claude Code chats", () => s.includeChats, v => { s.includeChats = v; void this.build(false); });
		const eng = row("Similarity").createEl("select");
		eng.createEl("option", { text: "Words (built-in)", value: "tfidf" });
		eng.createEl("option", { text: "Ollama embeddings", value: "ollama" });
		eng.value = s.engine;
		eng.onchange = () => { s.engine = eng.value as "tfidf" | "ollama"; void this.plugin.saveSettings(); void this.build(true); };
		const rb = panel.createEl("button", { text: "Rebuild layout", cls: "mb-rebuild" });
		rb.onclick = () => void this.build(true);
	}

	private syncSpin() {
		const on = this.plugin.settings.rotate;
		this.spinBtn.setText(on ? "⟳ Spinning" : "⟳ Still");
		this.spinBtn.toggleClass("is-on", on);
		this.spinBtn.setAttr("aria-label", on ? "Stop spinning" : "Start spinning");
		if (this.spinCheck) this.spinCheck.checked = on;
	}

	private status(t: string) { this.statusEl.setText(t); }

	private markDirty() {
		this.dirty++;
		this.statusEl.empty();
		const a = this.statusEl.createEl("a", { text: `${this.dirty} change${this.dirty > 1 ? "s" : ""}, refresh to show` });
		a.onclick = () => void this.build(false);
	}

	// ---------- three ----------
	private initThree() {
		const r = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
		r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		r.setClearColor(THEMES[this.plugin.settings.theme] ?? 0, 1);
		this.host.appendChild(r.domElement);
		this.renderer = r;
		this.scene.add(this.world);
		this.camera.position.set(0, 0.3, 6.2);
		const c = new OrbitControls(this.camera, r.domElement);
		c.enableDamping = true;
		c.dampingFactor = 0.08;
		c.minDistance = 0.8;
		c.maxDistance = 20;
		c.target.copy(this.targetCenter);
		this.controls = c;

		this.composer = new EffectComposer(r);
		this.composer.addPass(new RenderPass(this.scene, this.camera));
		this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.6, 0.35, 0.12);
		this.composer.addPass(this.bloom);

		r.domElement.addEventListener("webglcontextlost", e => e.preventDefault());
		r.domElement.addEventListener("pointerdown", e => { this.down = { x: e.clientX, y: e.clientY }; });
		r.domElement.addEventListener("pointerup", e => { void this.onClick(e); this.down = undefined; });
		r.domElement.addEventListener("pointermove", e => this.onMove(e));
		r.domElement.addEventListener("pointerleave", () => { this.hover = -1; this.tip.removeClass("is-on"); });

		this.ro = new ResizeObserver(() => this.resize());
		this.ro.observe(this.host);
		this.resize();
		const loop = () => { this.raf = window.requestAnimationFrame(loop); this.frame(); };
		loop();
	}

	private resize() {
		if (!this.renderer) return;
		const w = this.host.clientWidth, h = this.host.clientHeight;
		if (!w || !h) return;
		this.renderer.setSize(w, h, false);
		this.composer?.setSize(w, h);
		this.bloom?.resolution.set(w / 2, h / 2);
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		const pix = h * this.renderer.getPixelRatio() / 900;
		if (this.pointMat) this.pointMat.uniforms.uPix.value = pix;
		if (this.dustMat) this.dustMat.uniforms.uPix.value = pix;
	}

	private frame() {
		if (!this.renderer || !this.host.isShown() || document.hidden) return;
		const s = this.plugin.settings;
		if (this.flat) {
			const r = this.world.rotation.y, goal = Math.round(r / (Math.PI * 2)) * Math.PI * 2;
			this.world.rotation.y += (goal - r) * 0.08;
		} else if (s.rotate && !this.down && this.hover < 0) this.world.rotation.y += 0.0018 * s.rotateSpeed;
		this.dustAmt += ((this.form === "anatomy" ? 1 : 0) - this.dustAmt) * 0.08;
		if (this.dustMat) this.dustMat.uniforms.uMix.value = 1 - this.dustAmt;
		this.mix += (this.targetMix - this.mix) * 0.07;
		if (Math.abs(this.targetMix - this.mix) < 0.001) this.mix = this.targetMix;
		if (this.playing) {
			this.setTime(Math.min(1, this.time + 0.0022));
			this.slider.value = String(Math.round(this.time * 1000));
			if (this.time >= 1) this.togglePlay();
		}
		for (const m of [this.pointMat, this.lineMat]) if (m) m.uniforms.uMix.value = this.mix;
		this.controls!.target.lerp(this.targetCenter, 0.06);
		if (this.aimFront && this.zoomTo !== null) {
			// swing the camera to look straight at a flat chart
			const goal = this.controls!.target.clone().add(new THREE.Vector3(0, 0, this.zoomTo));
			this.camera.position.lerp(goal, 0.08);
			if (this.camera.position.distanceTo(goal) < 0.01) { this.aimFront = false; this.zoomTo = null; }
		} else
		if (this.zoomTo !== null) {
			const off = this.camera.position.clone().sub(this.controls!.target);
			const d = off.length();
			const nd = d + (this.zoomTo - d) * 0.06;
			this.camera.position.copy(this.controls!.target).add(off.setLength(nd));
			if (Math.abs(this.zoomTo - nd) < 0.01) this.zoomTo = null;
		}
		this.controls!.update();
		this.updateLabels();
		this.syncClear(s.bloom);
		if (s.bloom) this.composer!.render(); else this.renderer.render(this.scene, this.camera);
	}

	// ---------- data -> geometry ----------
	private async build(recompute: boolean) {
		this.dirty = 0;
		const s = this.plugin.settings;
		this.status("Reading vault…");
		let entries = await collectNotes(this.app, s);
		if (s.includeChats) {
			this.status("Reading Claude Code history…");
			await sleep(20);
			try { entries = entries.concat(collectChats()); } catch (e) { console.error("brainiac chats", e); }
		}
		if (!entries.length) { this.status("No notes found"); return; }

		const cache = this.plugin.layoutCache;
		const engineChanged = cache.engine !== s.engine;
		const missing = entries.filter(e => !cache.u[e.id]);
		cache.g = cache.g ?? {};
		if (recompute || engineChanged || Object.keys(cache.u).length === 0 || Object.keys(cache.g).length === 0 || missing.length > entries.length * 0.25) {
			let vecs: number[][];
			if (s.engine === "ollama") {
				try {
					vecs = await ollamaVectors(entries, s.ollamaModel, f => this.status(`Embedding with Ollama… ${Math.round(f * 100)}%`));
				} catch {
					new Notice("Brainiac: Ollama not reachable, using built-in word similarity");
					vecs = tfidfVectors(entries);
				}
			} else {
				this.status("Comparing notes…");
				await sleep(10);
				vecs = tfidfVectors(entries);
			}
			const { u, g } = await layout(vecs, entries, f => this.status(`Arranging the brain… ${Math.round(f * 100)}%`));
			cache.u = {};
			cache.g = {};
			entries.forEach((e, i) => { cache.u[e.id] = u[i]; cache.g![e.id] = g[i]; });
			cache.engine = s.engine;
			await this.plugin.saveSettings();
		}
		// notes added since the last layout get a stable spot in their folder's region
		for (const e of entries) {
			const r = rng(hashCode(e.id));
			if (!cache.u[e.id]) cache.u[e.id] = [r(), r(), r()];
			if (!cache.g[e.id]) cache.g[e.id] = [(r() - 0.5) * 3, (r() - 0.5) * 3, (r() - 0.5) * 3];
		}

		this.entries = entries;
		this.makeGeometry();
		const nNotes = entries.filter(e => e.kind === "note").length;
		const nChats = entries.length - nNotes;
		this.countText = `${nNotes.toLocaleString()} notes${nChats ? ` · ${nChats} chats` : ""} · ${(this.ownerOf.length - entries.length).toLocaleString()} paragraphs/messages`;
		this.status(this.countText);
	}

	private disposeScene() {
		for (const o of [this.points, this.dust, this.lines]) {
			if (!o) continue;
			this.world.remove(o);
			o.geometry.dispose();
			(o.material as THREE.Material).dispose();
		}
		this.points = this.dust = this.lines = undefined;
	}

	private makeGeometry() {
		this.disposeScene();
		const s = this.plugin.settings;
		const cache = this.plugin.layoutCache.u;
		const entries = this.entries;

		// groups sorted by size
		const counts = new Map<string, number>();
		for (const e of entries) counts.set(e.group, (counts.get(e.group) ?? 0) + 1);
		const names = [...counts.keys()].sort((a, b) => counts.get(b)! - counts.get(a)!);
		const dotCount = new Map<string, number>();
		for (const e of entries) dotCount.set(e.group, (dotCount.get(e.group) ?? 0) + e.parts.length + 1);
		this.groups = names.map(n => ({ name: n, count: counts.get(n)!, dots: dotCount.get(n)!, color: new THREE.Color() }));
		const gIndex = new Map(names.map((n, i) => [n, i]));

		// anatomy: folders split the cortex in proportion to their dots, then cerebellum/brainstem,
		// the rest share the nerve tree in slices
		type Slot = { snap?: Snapper; region?: Region; slice: number; of: number };
		const assign = new Map<string, Slot>();
		const folders = names.filter(n => n !== "Claude chats");
		// dots per group decide how much brain each one gets
		const dots = new Map<string, number>();
		for (const e of entries) dots.set(e.group, (dots.get(e.group) ?? 0) + e.parts.length + 1);
		const totalDots = folders.reduce((n, f) => n + dots.get(f)!, 0);
		// cortex: every folder with at least 0.4% of the dots; the rest go below
		const cortexGroups = folders.filter((f, i) => i < 10 && dots.get(f)! >= totalDots * 0.004);
		const rest = folders.filter(f => !cortexGroups.includes(f));
		const cortexDots = cortexGroups.reduce((n, f) => n + dots.get(f)!, 0);
		const pool = cortexPool(Math.round(cortexDots * 1.25) + 2000);
		const parts = partitionPool(pool.pts, cortexGroups.map(f => dots.get(f)!));
		cortexGroups.forEach((f, i) => assign.set(f, { snap: new Snapper(pool.pts, parts[i]), slice: 0, of: 1 }));
		const extras = EXTRA_REGIONS.filter(r => !(s.includeChats && r === "brainstem"));
		const shared: string[] = [];
		rest.forEach((f, i) => {
			if (i < extras.length) assign.set(f, { region: regionByName(extras[i]), slice: 0, of: 1 });
			else shared.push(f);
		});
		shared.forEach((n, j) => assign.set(n, { region: regionByName("nerves"), slice: j, of: shared.length }));
		if (counts.has("Claude chats")) assign.set("Claude chats", { region: regionByName("spine"), slice: 0, of: 1 });

		// chats run down the spine in time order
		const chatRank = new Map<string, number>();
		const chats = entries.filter(e => e.kind === "chat").sort((a, b) => a.time - b.time);
		chats.forEach((e, i) => chatRank.set(e.id, (i + 0.5) / chats.length));

		// time normalization, ignore silly early ctimes
		const times = entries.map(e => e.time).filter(t => t > Date.UTC(2015, 0, 1)).sort((a, b) => a - b);
		this.tMin = times[Math.floor(times.length * 0.01)] ?? 0;
		this.tMax = times[times.length - 1] ?? 1;
		const span = Math.max(1, this.tMax - this.tMin);


		const posA: number[] = [], birth: number[] = [], size: number[] = [], owner: number[] = [], part: number[] = [], rand: number[] = [];
		this.primaryIdx = [];
		this.uE = []; this.gE = []; this.gIdx = [];

		entries.forEach((e, ei) => {
			const gi = gIndex.get(e.group)!;
			const a = assign.get(e.group)!;
			const base = cache[e.id] as Vec3;
			this.uE.push(base); this.gE.push(this.plugin.layoutCache.g![e.id] as Vec3); this.gIdx.push(gi);
			let u: Vec3 = [base[0], base[1], base[2]];
			if (e.kind === "chat") u = [base[0], chatRank.get(e.id)!, base[2]];
			if (a.of > 1) u = [u[0], (a.slice + u[1]) / a.of, u[2]];
			const r = rng(hashCode(e.id));
			const b = Math.min(1, Math.max(0, (e.time - this.tMin) / span));
			const sats = e.parts.length;
			let anchor: Vec3 = [0, 0, 0];
			for (let k = 0; k <= sats; k++) {
				const j = k === 0 ? 0 : 0.035;
				const uk: Vec3 = [clamp01(u[0] + (r() - 0.5) * j), clamp01(u[1] + (r() - 0.5) * j), clamp01(u[2] + (r() - 0.5) * j)];
				let pa: Vec3;
				if (a.snap) {
					if (k === 0) anchor = a.snap.at(u);
					const t = k === 0 ? anchor : [anchor[0] + gauss(r) * 0.035, anchor[1] + gauss(r) * 0.035, anchor[2] + gauss(r) * 0.035] as Vec3;
					pa = a.snap.take(t[0], t[1], t[2]);
					if (k === 0) anchor = pa;
				} else pa = a.region!.sample(uk, r);
				if (k === 0) this.primaryIdx.push(owner.length);
				posA.push(...pa);
				// random offset inside a unit ball: where this dot sits around its note in other forms
				let ox = 0, oy = 0, oz = 0;
				if (k > 0) do { ox = r() * 2 - 1; oy = r() * 2 - 1; oz = r() * 2 - 1; } while (ox * ox + oy * oy + oz * oz > 1);
				rand.push(ox, oy, oz);
				birth.push(b + (k === 0 ? 0 : r() * 0.004));
				size.push(k === 0 ? 1.0 : 0.2 + r() * 0.12);
				owner.push(ei);
				part.push(k - 1);
			}
		});
		this.ownerOf = Int32Array.from(owner);
		this.partOf = Int16Array.from(part);
		this.partDim = Math.min(1, Math.max(0.1, 12000 / Math.max(1, part.length)));
		if (this.bloom) this.bloom.strength = part.length > 50000 ? 0.45 : 0.6;
		this.rawBirth = birth;
		this.births = birth.slice();
		this.anatomyPos = Float32Array.from(posA);
		this.pointRand = Float32Array.from(rand);

		const n = owner.length;
		const geo = new THREE.BufferGeometry();
		geo.setAttribute("position", new THREE.Float32BufferAttribute(posA, 3));
		geo.setAttribute("posA", new THREE.Float32BufferAttribute(posA, 3));
		geo.setAttribute("posB", new THREE.Float32BufferAttribute(posA, 3));
		geo.setAttribute("birth", new THREE.Float32BufferAttribute(birth, 1));
		geo.setAttribute("size", new THREE.Float32BufferAttribute(size, 1));
		geo.setAttribute("sel", new THREE.Float32BufferAttribute(new Float32Array(n), 1));
		geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(n * 3), 3));
		this.pointMat = new THREE.ShaderMaterial({
			vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
			uniforms: { uMix: { value: this.mix }, uTime: { value: this.time + 0.001 }, uSize: { value: s.pointSize * 22 }, uPix: { value: 1 }, uFocus: { value: 0 } },
		});
		this.points = new THREE.Points(geo, this.pointMat);
		this.points.frustumCulled = false;
		this.world.add(this.points);

		// outline dust
		const dgeo = new THREE.BufferGeometry();
		dgeo.setAttribute("position", new THREE.BufferAttribute(outlinePoints(9000), 3));
		this.dustMat = new THREE.ShaderMaterial({
			vertexShader: DUST_VERT, fragmentShader: DUST_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
			uniforms: { uMix: { value: this.mix }, uPix: { value: 1 }, uSize: { value: s.pointSize }, uColor: { value: new THREE.Color(0x8fa3c8) } },
		});
		this.dust = new THREE.Points(dgeo, this.dustMat);
		this.dust.frustumCulled = false;
		this.dust.visible = s.outline;
		this.world.add(this.dust);

		// links between primaries
		const idIndex = new Map(entries.map((e, i) => [e.id, i]));
		const lA: number[] = [], lB: number[] = [], lBirth: number[] = [];
		this.linkEnds = [];
		entries.forEach((e, i) => {
			for (const l of e.links) {
				const j = idIndex.get(l);
				if (j === undefined || j === i) continue;
				for (const k of [i, j]) {
					const p = this.primaryIdx[k];
					this.linkEnds.push(p);
					lA.push(posA[p * 3], posA[p * 3 + 1], posA[p * 3 + 2]);
					lB.push(posA[p * 3], posA[p * 3 + 1], posA[p * 3 + 2]);
				}
				const bb = Math.max(birth[this.primaryIdx[i]], birth[this.primaryIdx[j]]);
				lBirth.push(bb, bb);
			}
		});
		const lgeo = new THREE.BufferGeometry();
		lgeo.setAttribute("position", new THREE.Float32BufferAttribute(lA, 3));
		lgeo.setAttribute("posA", new THREE.Float32BufferAttribute(lA, 3));
		lgeo.setAttribute("posB", new THREE.Float32BufferAttribute(lB, 3));
		lgeo.setAttribute("birth", new THREE.Float32BufferAttribute(lBirth, 1));
		this.lineMat = new THREE.ShaderMaterial({
			vertexShader: LINE_VERT, fragmentShader: LINE_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
			uniforms: { uMix: { value: this.mix }, uTime: { value: this.time + 0.001 }, uColor: { value: new THREE.Color(0xbfd4ff) } },
		});
		this.lines = new THREE.LineSegments(lgeo, this.lineMat);
		this.lines.frustumCulled = false;
		this.lines.visible = s.links;
		this.world.add(this.lines);

		this.recolor();
		this.form = "";
		this.applyForm(s.layout in LAYOUTS || s.layout === "anatomy" ? s.layout : "anatomy", true);
		this.applyBirths();
		this.applyFocus();
		this.setTime(this.time);
		if (this.tlPanel.hasClass("is-open")) this.renderTimeline();
	}

	private recolor() {
		if (!this.points) return;
		const pal = PALETTES[this.plugin.settings.palette] ?? [];
		const n = this.groups.length;
		this.groups.forEach((g, i) => {
			if (!pal.length) g.color.setHSL((i * 0.618 + 0.02) % 1, 0.75, 0.6);
			else {
				const t = n === 1 ? 0.5 : 0.15 + 0.85 * (i / (n - 1));
				const x = t * (pal.length - 1), k = Math.floor(x);
				g.color.set(pal[k]).lerp(new THREE.Color(pal[Math.min(k + 1, pal.length - 1)]), x - k);
			}
		});
		const gi = new Map(this.groups.map((g, i) => [g.name, i]));
		const col = this.points.geometry.getAttribute("color") as THREE.BufferAttribute;
		const tmp = new THREE.Color();
		for (let p = 0; p < this.ownerOf.length; p++) {
			const g = this.groups[gi.get(this.entries[this.ownerOf[p]].group)!];
			const jitter = 0.75 + ((p * 2654435761) % 1000) / 1000 * 0.5;
			tmp.copy(g.color).multiplyScalar(jitter);
			col.setXYZ(p, tmp.r, tmp.g, tmp.b);
		}
		col.needsUpdate = true;
		this.renderLegend();
	}

	private renderLegend() {
		this.legend.empty();
		this.legend.createDiv({ text: "BY FOLDER", cls: "mb-legend-title" });
		const wrap = this.legend.createDiv("mb-legend-items");
		for (const g of this.groups) {
			const it = wrap.createDiv("mb-legend-item");
			if (this.focusGroup === g.name) it.addClass("is-on");
			const dot = it.createSpan("mb-dot");
			dot.style.background = `#${g.color.getHexString()}`;
			it.createSpan({ text: g.name });
			it.createSpan({ text: ` ${g.count}`, cls: "mb-count" });
			it.onclick = () => this.focusFolder(this.focusGroup === g.name ? null : g.name);
		}
	}

	// ---------- forms ----------
	private ctx(): Ctx {
		return { entries: this.entries, gi: this.gIdx, groups: this.groups, u: this.uE, g: this.gE, tMin: this.tMin, tMax: this.tMax };
	}

	// per-point positions for a form: anatomy is precomputed, the rest scatter
	// each note's dots around the note's anchor
	private formPositions(form: string): { pos: Float32Array; labels: Label[]; flat: boolean; stat?: string } {
		if (form === "anatomy" || !LAYOUTS[form]) return { pos: this.anatomyPos!, labels: [], flat: false };
		const res = LAYOUTS[form](this.ctx());
		const n = this.ownerOf.length, pos = new Float32Array(n * 3), rnd = this.pointRand!;
		const zf = res.flat ? 0.15 : 1;
		for (let p = 0; p < n; p++) {
			const e = this.ownerOf[p], sp = res.spread[e];
			pos[p * 3] = res.anchor[e * 3] + rnd[p * 3] * sp;
			pos[p * 3 + 1] = res.anchor[e * 3 + 1] + rnd[p * 3 + 1] * sp;
			pos[p * 3 + 2] = res.anchor[e * 3 + 2] + rnd[p * 3 + 2] * sp * zf;
		}
		return { pos, labels: res.labels, flat: res.flat, stat: res.stat };
	}

	private applyForm(form: string, instant = false) {
		if (!this.points || form === this.form) return;
		this.form = form;
		this.formBtns.forEach((b, id) => b.toggleClass("is-on", id === form));
		const { pos, labels, flat, stat } = this.formPositions(form);
		const A = this.points.geometry.getAttribute("posA") as THREE.BufferAttribute;
		const B = this.points.geometry.getAttribute("posB") as THREE.BufferAttribute;
		const a = A.array as Float32Array, b = B.array as Float32Array, t = this.mix;
		// morph from wherever the dots are right now
		for (let i = 0; i < a.length; i++) a[i] = instant ? pos[i] : a[i] + (b[i] - a[i]) * t;
		b.set(pos);
		A.needsUpdate = B.needsUpdate = true;
		if (this.lines) {
			const LA = this.lines.geometry.getAttribute("posA") as THREE.BufferAttribute;
			const LB = this.lines.geometry.getAttribute("posB") as THREE.BufferAttribute;
			const la = LA.array as Float32Array, lb = LB.array as Float32Array;
			const ends = this.linkEnds;
			for (let i = 0; i < ends.length; i++) {
				const p = ends[i];
				for (let k = 0; k < 3; k++) {
					la[i * 3 + k] = instant ? pos[p * 3 + k] : la[i * 3 + k] + (lb[i * 3 + k] - la[i * 3 + k]) * t;
					lb[i * 3 + k] = pos[p * 3 + k];
				}
			}
			LA.needsUpdate = LB.needsUpdate = true;
		}
		this.mix = instant ? 1 : 0;
		this.targetMix = 1;
		this.flat = flat;
		this.statEl.setText(stat ?? "");

		// labels
		for (const fl of this.formLabels) fl.el.remove();
		this.formLabels = labels.map(l => {
			const el = this.labelLayer.createDiv({ text: l.text, cls: `mb-label mb-label-${l.kind}` });
			if (l.kind === "group" && this.groups.some(g => g.name === l.text)) el.onclick = () => this.focusFolder(l.text);
			return { l, el };
		});

		// frame the camera on the new form
		let mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
		for (const p of this.primaryIdx) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], pos[p * 3 + k]); mx[k] = Math.max(mx[k], pos[p * 3 + k]); }
		for (const fl of labels) for (let k = 0; k < 2; k++) { mn[k] = Math.min(mn[k], fl.p[k]); mx[k] = Math.max(mx[k], fl.p[k]); }
		// right-aligned row labels hang off the left edge; leave room for their text
		const longest = Math.max(0, ...labels.filter(l => l.kind === "axis" && l.text.length > 6).map(l => l.text.length));
		if (flat && longest && form !== "folders") mn[0] -= longest * 0.05;
		if (form === "anatomy") { mn = [-1, this.plugin.settings.includeChats ? -3.6 : -2.6, -1]; mx = [1, 0.9, 1]; }
		this.targetCenter.set((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2 + (flat ? (mx[1] - mn[1]) * 0.04 : 0), flat ? 0 : (mn[2] + mx[2]) / 2);
		const halfH = (mx[1] - mn[1]) / 2, halfW = (mx[0] - mn[0]) / 2;
		const fit = Math.max(halfH, halfW / Math.max(0.5, this.camera.aspect)) / Math.tan((this.camera.fov / 2) * Math.PI / 180);
		this.zoomTo = Math.min(18, Math.max(2.5, fit * (flat ? 1.32 : 1.25) + (flat ? 0 : (mx[2] - mn[2]) / 2)));
		this.aimFront = flat;
		if (instant) {
			this.controls?.target.copy(this.targetCenter);
			this.camera.position.copy(this.targetCenter).add(new THREE.Vector3(0, flat ? 0 : 0.3, this.zoomTo));
			this.zoomTo = null; this.aimFront = false;
		}
		if (this.focusGroup) this.drawTrail();
	}

	private updateLabels() {
		if (!this.formLabels.length || !this.renderer) return;
		const w = this.host.clientWidth, h = this.host.clientHeight;
		const v = new THREE.Vector3();
		const op = String(Math.max(0, (this.mix - 0.6) / 0.4));
		for (const { l, el } of this.formLabels) {
			v.set(l.p[0], l.p[1], l.p[2]).applyMatrix4(this.world.matrixWorld).project(this.camera);
			const vis = v.z < 1 && Math.abs(v.x) < 1.15 && Math.abs(v.y) < 1.15;
			el.style.opacity = vis ? op : "0";
			const x = (v.x * 0.5 + 0.5) * w, y = (-v.y * 0.5 + 0.5) * h;
			el.style.transform = l.kind === "axis" && l.text.length > 6 && this.form !== "folders"
				? `translate(${x}px, ${y}px) translate(-100%, -50%)`
				: `translate(${x}px, ${y}px) translate(-50%, ${l.kind === "group" ? "-150%" : "-50%"})`;
		}
	}

	private applyBirths() {
		if (!this.points) return;
		const nf = this.plugin.settings.newestFirst;
		const attr = this.points.geometry.getAttribute("birth") as THREE.BufferAttribute;
		for (let p = 0; p < this.rawBirth.length; p++) {
			const b = nf ? 1 - this.rawBirth[p] : this.rawBirth[p];
			this.births[p] = b;
			attr.setX(p, b);
		}
		attr.needsUpdate = true;
		if (this.lines) {
			const la = this.lines.geometry.getAttribute("birth") as THREE.BufferAttribute;
			for (let i = 0; i < this.linkEnds.length; i += 2) {
				const bb = Math.max(this.births[this.linkEnds[i]], this.births[this.linkEnds[i + 1]]);
				la.setX(i, bb); la.setX(i + 1, bb);
			}
			la.needsUpdate = true;
		}
	}

	private applyTheme() {
		const c = THEMES[this.plugin.settings.theme] ?? 0;
		this.contentEl.style.background = `#${c.toString(16).padStart(6, "0")}`;
		this.clearFor = -1; // re-set on next frame for the current glow mode
	}

	// the glow pipeline sRGB-encodes the background once more, so pre-compensate
	private clearFor = -1;
	private syncClear(bloom: boolean) {
		const key = (bloom ? 1 : 0) + (THEMES[this.plugin.settings.theme] ?? 0) * 2;
		if (key === this.clearFor || !this.renderer) return;
		this.clearFor = key;
		const col = new THREE.Color(THEMES[this.plugin.settings.theme] ?? 0);
		if (bloom) col.convertSRGBToLinear();
		this.renderer.setClearColor(col, 1);
	}

	// ---------- folder hub: trail through its notes in time order + list ----------
	private focusFolder(name: string | null) {
		this.focusGroup = name;
		this.focusMonth = null;
		this.searchEl.value = "";
		this.applyFocus();
		this.renderLegend();
		this.drawTrail();
		this.renderHub();
	}

	private drawTrail() {
		if (this.trail) { this.world.remove(this.trail); this.trail.geometry.dispose(); (this.trail.material as THREE.Material).dispose(); this.trail = undefined; }
		if (!this.focusGroup || !this.points) return;
		const idx = this.entries.map((e, i) => i).filter(i => this.entries[i].group === this.focusGroup).sort((a, b) => this.entries[a].time - this.entries[b].time);
		if (idx.length < 2) return;
		// thin the path on huge folders so it stays readable
		const step = Math.max(1, Math.floor(idx.length / 400));
		const B = (this.points.geometry.getAttribute("posB") as THREE.BufferAttribute).array as Float32Array;
		const pts: number[] = [], cols: number[] = [];
		const c = new THREE.Color();
		for (let j = 0; j < idx.length; j += step) {
			const p = this.primaryIdx[idx[j]];
			pts.push(B[p * 3], B[p * 3 + 1], B[p * 3 + 2]);
			c.setHSL(0.62 - 0.62 * (j / idx.length), 0.9, 0.6);
			cols.push(c.r, c.g, c.b);
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
		geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
		this.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
		this.trail.frustumCulled = false;
		this.world.add(this.trail);
	}

	private renderHub() {
		this.hubPanel.empty();
		if (!this.focusGroup) { this.hubPanel.removeClass("is-open"); return; }
		this.hubPanel.addClass("is-open");
		const list = this.entries.filter(e => e.group === this.focusGroup).sort((a, b) => b.time - a.time);
		const head = this.hubPanel.createDiv("mb-hub-head");
		head.createDiv({ text: this.focusGroup, cls: "mb-hub-title" });
		const close = head.createEl("button", { text: "×", cls: "mb-hub-close" });
		close.onclick = () => this.focusFolder(null);
		const first = list[list.length - 1], last = list[0];
		const dots = list.reduce((n, e) => n + e.parts.length + 1, 0);
		this.hubPanel.createDiv({ text: `${list.length} ${list[0]?.kind === "chat" ? "chats" : "notes"} · ${dots.toLocaleString()} dots · ${new Date(first.time).toLocaleDateString()} – ${new Date(last.time).toLocaleDateString()}`, cls: "mb-hub-meta" });
		this.hubPanel.createDiv({ text: "The trail runs oldest (blue) to newest (red). Newest first:", cls: "mb-hub-meta" });
		const ul = this.hubPanel.createDiv("mb-hub-list");
		for (const e of list.slice(0, 300)) {
			const it = ul.createDiv("mb-hub-item");
			it.createDiv({ text: e.title, cls: "mb-hub-name" });
			it.createDiv({ text: `${new Date(e.time).toLocaleDateString()} · ${e.parts.length} ${e.kind === "chat" ? "messages" : "paragraphs"}`, cls: "mb-hub-sub" });
			it.onclick = async () => {
				if (e.kind === "chat") { await navigator.clipboard.writeText(`claude --resume ${e.resume}`); new Notice(`Copied: claude --resume ${e.resume}`); return; }
				const f = this.app.vault.getAbstractFileByPath(e.id);
				if (f instanceof TFile) await this.app.workspace.getLeaf("tab").openFile(f);
			};
		}
		if (list.length > 300) ul.createDiv({ text: `+ ${list.length - 300} more`, cls: "mb-hub-sub" });
	}

	// ---------- "When you wrote": dots per month, click a bar to light it up ----------
	private renderTimeline() {
		const el = this.tlPanel;
		el.empty();
		if (!el.hasClass("is-open") || !this.entries.length) return;
		const head = el.createDiv("mb-hub-head");
		head.createDiv({ text: "When you wrote", cls: "mb-hub-title" });
		const close = head.createEl("button", { text: "×", cls: "mb-hub-close" });
		close.onclick = () => { el.removeClass("is-open"); };
		el.createDiv({ text: "Dots per month: notes and their paragraphs vs. messages you typed to Claude. Click a month to light it up in the brain.", cls: "mb-hub-meta" });
		const mk = (t: number) => { const d = new Date(t); return d.getFullYear() * 12 + d.getMonth(); };
		const m0 = mk(this.tMin), m1 = mk(this.tMax), M = Math.max(1, m1 - m0 + 1);
		const notes = new Array<number>(M).fill(0), chats = new Array<number>(M).fill(0);
		for (const e of this.entries) {
			const m = mk(e.time) - m0;
			if (m < 0 || m >= M) continue;
			(e.kind === "chat" ? chats : notes)[m] += e.parts.length + 1;
		}
		const max = Math.max(1, ...notes.map((v, i) => v + chats[i]));
		const W = 300, H = 140;
		const cv = el.createEl("canvas", { cls: "mb-tl-canvas" });
		cv.width = W * 2; cv.height = H * 2; cv.style.width = W + "px"; cv.style.height = H + "px";
		const g = cv.getContext("2d")!;
		g.scale(2, 2);
		const bw = W / M;
		const noteC = "#7fd1a0", chatC = "#8fa8ff";
		for (let m = 0; m < M; m++) {
			const hn = notes[m] / max * (H - 16), hc = chats[m] / max * (H - 16);
			const on = this.focusMonth === null || this.focusMonth === m0 + m;
			g.globalAlpha = on ? 1 : 0.3;
			g.fillStyle = noteC; g.fillRect(m * bw + 0.5, H - 14 - hn, Math.max(1, bw - 1), hn);
			g.fillStyle = chatC; g.fillRect(m * bw + 0.5, H - 14 - hn - hc, Math.max(1, bw - 1), hc);
		}
		g.globalAlpha = 1; g.fillStyle = "#8b93a5"; g.font = "10px sans-serif";
		const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
		const every = Math.max(1, Math.ceil(M / 6));
		for (let m = 0; m < M; m += every) { const k = m0 + m; g.fillText(`${MON[k % 12]} ${String(Math.floor(k / 12)).slice(2)}`, m * bw, H - 2); }
		cv.onclick = ev => {
			const m = Math.floor((ev.offsetX / W) * M);
			this.focusMonth = this.focusMonth === m0 + m ? null : m0 + m;
			this.focusGroup = null; this.searchEl.value = "";
			this.applyFocus(); this.renderLegend(); this.drawTrail(); this.renderHub(); this.renderTimeline();
		};
		const key = el.createDiv("mb-tl-key");
		key.createSpan({ text: "■ notes + paragraphs  ", attr: { style: `color:${noteC}` } });
		key.createSpan({ text: "■ chat messages", attr: { style: `color:${chatC}` } });
		if (this.focusMonth !== null) {
			const k = this.focusMonth, n = notes[k - m0] + chats[k - m0];
			const sel = el.createDiv("mb-hub-meta");
			sel.setText(`${n.toLocaleString()} dots in ${MON[k % 12]} ${Math.floor(k / 12)} · `);
			const clr = sel.createEl("a", { text: "Clear" });
			clr.onclick = () => { this.focusMonth = null; this.applyFocus(); this.renderTimeline(); };
		}
		// days written + longest run
		const days = [...new Set(this.entries.map(e => { const d = new Date(e.time); d.setHours(0, 0, 0, 0); return d.getTime(); }))].sort((a, b) => a - b);
		let run = 1, best = 1;
		for (let i = 1; i < days.length; i++) { run = Math.round((days[i] - days[i - 1]) / 864e5) === 1 ? run + 1 : 1; best = Math.max(best, run); }
		el.createDiv({ text: `Written on ${days.length} days · longest run ${best} days in a row`, cls: "mb-hub-meta" });
	}

	private applyFocus() {
		if (!this.points || !this.pointMat) return;
		const q = this.searchEl.value.trim().toLowerCase();
		const sel = this.points.geometry.getAttribute("sel") as THREE.BufferAttribute;
		const active = !!q || !!this.focusGroup || this.focusMonth !== null;
		const fm = this.focusMonth;
		const match = this.entries.map(e => {
			if (q) return e.title.toLowerCase().includes(q);
			if (fm !== null) { const d = new Date(e.time); return d.getFullYear() * 12 + d.getMonth() === fm; }
			return e.group === this.focusGroup;
		});
		for (let p = 0; p < this.ownerOf.length; p++) sel.setX(p, active && match[this.ownerOf[p]] ? 1 : 0);
		sel.needsUpdate = true;
		this.pointMat.uniforms.uFocus.value = active ? 1 : 0;
		const hits = match.filter(Boolean).length;
		this.status(q ? `${hits} match${hits === 1 ? "" : "es"}` : this.countText);
	}

	private setTime(t: number) {
		this.time = t;
		const eps = t >= 1 ? 0.01 : 0;
		if (this.pointMat) this.pointMat.uniforms.uTime.value = t + eps;
		if (this.lineMat) this.lineMat.uniforms.uTime.value = t + eps;
		const d = new Date(this.tMin + (this.tMax - this.tMin) * t);
		this.dateEl.setText(d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }));
	}

	private togglePlay() {
		this.playing = !this.playing;
		if (this.playing && this.time >= 1) this.setTime(0);
		this.playBtn.setText(this.playing ? "❚❚" : "▶");
	}

	// ---------- picking ----------
	// returns a point index; every point is a real note, paragraph or message
	private pick(e: PointerEvent): number {
		if (!this.points || !this.pointMat) return -1;
		const rect = this.host.getBoundingClientRect();
		const mx = e.clientX - rect.left, my = e.clientY - rect.top;
		const posA = (this.points.geometry.getAttribute("posA") as THREE.BufferAttribute).array;
		const posB = (this.points.geometry.getAttribute("posB") as THREE.BufferAttribute).array;
		const sel = (this.points.geometry.getAttribute("sel") as THREE.BufferAttribute).array;
		const focus = this.pointMat.uniforms.uFocus.value > 0;
		const m = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse).multiply(this.world.matrixWorld);
		const me = m.elements, t = this.mix;
		let best = -1, bestD = 10 * 10, bestW = Infinity;
		for (let p = 0; p < this.ownerOf.length; p++) {
			if (this.births[p] > this.time + 0.001) continue;
			if (focus && sel[p] < 0.5) continue;
			const x = posA[p * 3] + (posB[p * 3] - posA[p * 3]) * t;
			const y = posA[p * 3 + 1] + (posB[p * 3 + 1] - posA[p * 3 + 1]) * t;
			const z = posA[p * 3 + 2] + (posB[p * 3 + 2] - posA[p * 3 + 2]) * t;
			const w = me[3] * x + me[7] * y + me[11] * z + me[15];
			if (w <= 0) continue;
			const sx = ((me[0] * x + me[4] * y + me[8] * z + me[12]) / w * 0.5 + 0.5) * rect.width;
			const sy = (-(me[1] * x + me[5] * y + me[9] * z + me[13]) / w * 0.5 + 0.5) * rect.height;
			// main dots are bigger, so give them a larger hit radius
			const d = ((sx - mx) ** 2 + (sy - my) ** 2) / (this.partOf[p] < 0 ? 2.2 : 1);
			if (d < bestD || (d < bestD + 6 && w < bestW)) { best = p; bestD = d; bestW = w; }
		}
		return best;
	}

	private moveQueued?: PointerEvent;
	private onMove(e: PointerEvent) {
		if (e.buttons) { this.tip.removeClass("is-on"); return; }
		if (!this.moveQueued) window.requestAnimationFrame(() => { const ev = this.moveQueued!; this.moveQueued = undefined; this.hoverAt(ev); });
		this.moveQueued = e;
	}

	private hoverAt(e: PointerEvent) {
		const p = this.pick(e);
		this.hover = p;
		if (p < 0) { this.tip.removeClass("is-on"); this.renderer!.domElement.removeClass("is-pointer"); return; }
		const en = this.entries[this.ownerOf[p]];
		const part = this.partOf[p];
		const rect = this.host.getBoundingClientRect();
		this.tip.empty();
		this.tip.createDiv({ text: en.title, cls: "mb-tip-title" });
		if (part >= 0) {
			const label = en.kind === "chat" ? en.parts[part] : `paragraph ${part + 1} of ${en.parts.length}${en.parts[part] ? ` · § ${en.parts[part]}` : ""}`;
			this.tip.createDiv({ text: label, cls: "mb-tip-meta" });
		} else {
			this.tip.createDiv({ text: `${en.parts.length} ${en.kind === "chat" ? "messages" : "paragraphs"}`, cls: "mb-tip-meta" });
		}
		this.tip.createDiv({ text: `${en.group} · ${new Date(en.time).toLocaleDateString()}`, cls: "mb-tip-meta" });
		if (en.kind === "chat") this.tip.createDiv({ text: "click to copy resume command", cls: "mb-tip-meta" });
		this.tip.style.transform = `translate(${e.clientX - rect.left + 14}px, ${e.clientY - rect.top + 14}px)`;
		this.tip.addClass("is-on");
		this.renderer!.domElement.addClass("is-pointer");
	}

	private async onClick(e: PointerEvent) {
		if (!this.down || Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > 4) return;
		const p = this.pick(e);
		if (p < 0) return;
		const en = this.entries[this.ownerOf[p]];
		if (en.kind === "chat") {
			await navigator.clipboard.writeText(`claude --resume ${en.resume}`);
			new Notice(`Copied: claude --resume ${en.resume}`);
			return;
		}
		const f = this.app.vault.getAbstractFileByPath(en.id);
		if (!(f instanceof TFile)) return;
		const leaf = this.app.workspace.getLeaf(Keymap.isModEvent(e) || "tab");
		await leaf.openFile(f);
		// jump to the paragraph's heading when there is one
		const h = this.partOf[p] >= 0 ? en.parts[this.partOf[p]] : "";
		if (h) await this.app.workspace.openLinkText(`${f.path}#${h}`, "", false);
	}
}

function gauss(r: () => number) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }
function clamp01(x: number) { return Math.min(0.999, Math.max(0.001, x)); }
function hashCode(s: string) { let h = 7; for (let i = 0; i < s.length; i++) h = Math.imul(h, 31) + s.charCodeAt(i) | 0; return h >>> 0; }
function sleep(ms: number) { return new Promise(r => window.setTimeout(r, ms)); }
