// Procedural anatomy: point samplers for brain regions, brainstem, spine, nerves.
// Coordinates: y up, z forward (face), x left-right. Brain spans roughly [-1,1].

export type Vec3 = [number, number, number];

// small deterministic RNG so the brain looks the same every open
export function rng(seed: number) {
	let s = seed >>> 0 || 1;
	return () => {
		s ^= s << 13; s >>>= 0;
		s ^= s >> 17;
		s ^= s << 5; s >>>= 0;
		return s / 4294967296;
	};
}

function gauss(r: () => number) {
	return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
}

// cheap 3D value noise for gyri bumps
function hash(x: number, y: number, z: number) {
	const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
	return h - Math.floor(h);
}
function noise(x: number, y: number, z: number) {
	const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
	const xf = x - xi, yf = y - yi, zf = z - zi;
	const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf), w = zf * zf * (3 - 2 * zf);
	let acc = 0;
	for (let dx = 0; dx < 2; dx++) for (let dy = 0; dy < 2; dy++) for (let dz = 0; dz < 2; dz++) {
		const wgt = (dx ? u : 1 - u) * (dy ? v : 1 - v) * (dz ? w : 1 - w);
		acc += wgt * hash(xi + dx, yi + dy, zi + dz);
	}
	return acc;
}
function gyri(x: number, y: number, z: number) {
	return 0.6 * noise(x * 6, y * 6, z * 6) + 0.3 * noise(x * 13, y * 13, z * 13) + 0.1 * noise(x * 27, y * 27, z * 27);
}

export interface Region {
	name: string;
	// returns a point inside the region; u in [0,1]^3 lets callers place
	// semantically-close notes in nearby spots
	sample(u: Vec3, r: () => number): Vec3;
}

const cerebellum: Region = {
	name: "cerebellum",
	sample(u, r) {
		const a = (u[0] - 0.5) * Math.PI * 1.2;
		const b = (u[1] - 0.5) * Math.PI * 0.9;
		const d = 1 - Math.pow(r(), 2) * 0.5;
		const fold = 1 + 0.12 * Math.sin(b * 22);
		return [
			Math.sin(a) * 0.48 * d,
			-0.32 + Math.sin(b) * 0.2 * d * fold,
			-0.62 + Math.cos(a) * Math.cos(b) * 0.22 * d,
		];
	},
};

// brainstem + spinal cord as a curve going down
function spinePoint(t: number): Vec3 {
	// t=0 top of brainstem, t=1 bottom of spine
	const y = -0.15 - t * 3.4;
	const z = -0.25 - 0.12 * Math.sin(t * Math.PI * 1.6) + 0.05 * t;
	return [0, y, z];
}

const brainstem: Region = {
	name: "brainstem",
	sample(u, r) {
		const t = u[1] * 0.18;
		const c = spinePoint(t);
		const rad = 0.14 * (1 - u[1] * 0.3) * Math.sqrt(r());
		const ang = u[0] * Math.PI * 2;
		return [c[0] + Math.cos(ang) * rad, c[1], c[2] + Math.sin(ang) * rad];
	},
};

const spine: Region = {
	name: "spine",
	sample(u, r) {
		const t = 0.18 + u[1] * 0.82;
		const c = spinePoint(t);
		const rad = (0.07 - 0.03 * t) * Math.sqrt(r());
		const ang = u[0] * Math.PI * 2;
		return [c[0] + Math.cos(ang) * rad, c[1], c[2] + Math.sin(ang) * rad];
	},
};

// peripheral nerves fanning from the spine into a torso silhouette
const nerves: Region = {
	name: "nerves",
	sample(u, r) {
		const t = 0.2 + u[1] * 0.75;
		const c = spinePoint(t);
		const side = u[0] < 0.5 ? -1 : 1;
		const along = Math.pow(r(), 0.8);
		// torso half-width: shoulders wide, waist narrower, hips out
		const width = 0.55 + 0.9 * Math.exp(-Math.pow((t - 0.3) / 0.18, 2)) + 0.35 * Math.exp(-Math.pow((t - 0.92) / 0.1, 2));
		const droop = along * along * (0.25 + 0.2 * t);
		return [
			side * along * width + gauss(r) * 0.015,
			c[1] - droop + gauss(r) * 0.01,
			c[2] + 0.15 * along + gauss(r) * 0.02 + u[2] * 0.15,
		];
	},
};

// cortex: two hemispheres, sampled once into a point pool labelled by lobe.
// notes are snapped onto pool points so they fill the real shape.
export const LOBES = ["frontal", "prefrontal", "parietal", "temporal", "occipital"] as const;

function lobeOf(x: number, y: number, z: number): string {
	if (z > 0.55) return "prefrontal";
	if (y < 0.12 && z > -0.5) return "temporal";
	if (z > 0.02) return "frontal";
	if (z > -0.5) return "parietal";
	return "occipital";
}

export interface Pool { pts: Float32Array; lobe: string[]; }

export function cortexPool(count: number, seed = 3): Pool {
	const r = rng(seed);
	const pts = new Float32Array(count * 3);
	const lobe: string[] = [];
	let i = 0;
	while (i < count) {
		// random direction, radius biased toward the surface
		const z0 = r() * 2 - 1, th = r() * Math.PI * 2, s0 = Math.sqrt(1 - z0 * z0);
		const dx = s0 * Math.cos(th), dy = z0, dz = s0 * Math.sin(th);
		if (dy < -0.8) continue; // slightly flattened underside
		const surf = 1 + 0.09 * (gyri(dx * 1.3, dy * 1.3, dz * 1.3) - 0.5);
		const rad = surf * (1 - Math.pow(r(), 3) * 0.55);
		const side = dx < 0 ? -1 : 1;
		const x = side * (0.03 + Math.abs(dx) * 0.6 * rad);
		const y = 0.22 + dy * 0.62 * rad;
		const z = dz * 0.95 * rad;
		pts[i * 3] = x; pts[i * 3 + 1] = y; pts[i * 3 + 2] = z;
		lobe.push(lobeOf(x, y, z));
		i++;
	}
	return { pts, lobe };
}

// split pool indices into contiguous chunks along a Morton curve, sized by weight,
// so each group gets one compact patch of cortex
export function partitionPool(pts: Float32Array, weights: number[]): number[][] {
	const n = pts.length / 3;
	const keys = new Array<[number, number]>(n);
	const spread = (v: number) => {
		let x = v & 0x3ff;
		x = (x | (x << 16)) & 0x30000ff; x = (x | (x << 8)) & 0x300f00f;
		x = (x | (x << 4)) & 0x30c30c3; x = (x | (x << 2)) & 0x9249249;
		return x;
	};
	for (let i = 0; i < n; i++) {
		// hemisphere first so patches do not straddle the fissure
		const side = pts[i * 3] < 0 ? 0 : 1;
		const qx = Math.floor((Math.abs(pts[i * 3]) / 0.7) * 1023);
		const qy = Math.floor(((pts[i * 3 + 1] + 0.5) / 1.4) * 1023);
		const qz = Math.floor(((pts[i * 3 + 2] + 1) / 2) * 1023);
		keys[i] = [side * 2 ** 34 + spread(qz) * 4 + spread(qy) * 2 + spread(qx), i];
	}
	keys.sort((a, b) => a[0] - b[0]);
	const total = weights.reduce((a, b) => a + b, 0) || 1;
	const out: number[][] = [];
	let at = 0;
	weights.forEach((w, gi) => {
		const take = gi === weights.length - 1 ? n - at : Math.round((w / total) * n);
		out.push(keys.slice(at, at + take).map(k => k[1]));
		at += take;
	});
	return out;
}

// uniform grid for nearest-free-point lookup
export class Snapper {
	private cells = new Map<string, number[]>();
	private used: Uint8Array;
	min: Vec3 = [Infinity, Infinity, Infinity];
	max: Vec3 = [-Infinity, -Infinity, -Infinity];
	constructor(private pts: Float32Array, private idx: number[], private cell = 0.035) {
		this.used = new Uint8Array(pts.length / 3);
		for (const i of idx) {
			const x = pts[i * 3], y = pts[i * 3 + 1], z = pts[i * 3 + 2];
			for (let a = 0; a < 3; a++) { this.min[a] = Math.min(this.min[a], pts[i * 3 + a]); this.max[a] = Math.max(this.max[a], pts[i * 3 + a]); }
			const k = this.key(x, y, z);
			let c = this.cells.get(k);
			if (!c) this.cells.set(k, c = []);
			c.push(i);
		}
	}
	get size() { return this.idx.length; }
	private key(x: number, y: number, z: number) {
		return `${Math.floor(x / this.cell)},${Math.floor(y / this.cell)},${Math.floor(z / this.cell)}`;
	}
	// nearest unused pool point to (x,y,z); reuses points once the pool runs dry
	take(x: number, y: number, z: number): Vec3 {
		const cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell), cz = Math.floor(z / this.cell);
		for (let ring = 0; ring < 40; ring++) {
			let best = -1, bestD = Infinity;
			for (let a = -ring; a <= ring; a++) for (let b = -ring; b <= ring; b++) for (let c = -ring; c <= ring; c++) {
				if (Math.max(Math.abs(a), Math.abs(b), Math.abs(c)) !== ring) continue;
				const list = this.cells.get(`${cx + a},${cy + b},${cz + c}`);
				if (!list) continue;
				for (const i of list) {
					if (this.used[i]) continue;
					const d = (this.pts[i * 3] - x) ** 2 + (this.pts[i * 3 + 1] - y) ** 2 + (this.pts[i * 3 + 2] - z) ** 2;
					if (d < bestD) { bestD = d; best = i; }
				}
			}
			if (best >= 0) {
				this.used[best] = 1;
				return [this.pts[best * 3], this.pts[best * 3 + 1], this.pts[best * 3 + 2]];
			}
		}
		this.used.fill(0); // exhausted: allow reuse
		const i = this.idx[Math.floor(Math.random() * this.idx.length)];
		return [this.pts[i * 3] + (Math.random() - 0.5) * 0.02, this.pts[i * 3 + 1] + (Math.random() - 0.5) * 0.02, this.pts[i * 3 + 2] + (Math.random() - 0.5) * 0.02];
	}
	// map a unit-cube coordinate into this region's bounding box
	at(u: Vec3): Vec3 {
		return [0, 1, 2].map(a => this.min[a] + u[a] * (this.max[a] - this.min[a])) as Vec3;
	}
}

export const REGIONS: Region[] = [cerebellum, brainstem, spine, nerves];

export function regionByName(n: string) {
	return REGIONS.find(r => r.name === n)!;
}

// faint outline dust so the shape reads even with few notes
export function outlinePoints(count: number, seed = 7): Float32Array {
	const r = rng(seed);
	const cortex = cortexPool(Math.floor(count * 0.55), seed + 1).pts;
	const out = new Float32Array(count * 3);
	out.set(cortex, 0);
	const rest = [cerebellum, brainstem, spine, spine, nerves, nerves];
	for (let i = cortex.length / 3; i < count; i++) {
		const p = rest[Math.floor(r() * rest.length)].sample([r(), r(), r()], r);
		out.set(p, i * 3);
	}
	return out;
}

// sphere layout: one ball per group arranged on a ring-ish spiral
// sphere layout: biggest group in the middle, the rest on a sunflower spiral
export function sphereCenter(i: number, n: number): Vec3 {
	if (i === 0) return [0, 0, 0];
	const golden = Math.PI * (3 - Math.sqrt(5));
	const r = 0.75 + 0.62 * Math.sqrt(i);
	const th = golden * i;
	return [Math.cos(th) * r, Math.sin(th) * r * 0.8, (i % 2 ? 0.25 : -0.25)];
}

export function spherePoint(center: Vec3, radius: number, u: Vec3, r: () => number): Vec3 {
	const x = u[0] * 2 - 1, y = u[1] * 2 - 1, z = u[2] * 2 - 1;
	const sx = x * Math.sqrt(1 - y * y / 2 - z * z / 2 + y * y * z * z / 3);
	const sy = y * Math.sqrt(1 - z * z / 2 - x * x / 2 + z * z * x * x / 3);
	const sz = z * Math.sqrt(1 - x * x / 2 - y * y / 2 + x * x * y * y / 3);
	const j = 0.04 * radius;
	return [center[0] + sx * radius + (r() - 0.5) * j, center[1] + sy * radius + (r() - 0.5) * j, center[2] + sz * radius + (r() - 0.5) * j];
}
