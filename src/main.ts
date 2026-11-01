import { Plugin } from "obsidian";
import { BrainView, VIEW_TYPE } from "./view";
import { BrainSettings, DEFAULT_SETTINGS } from "./data";

interface LayoutCache { engine: string; u: Record<string, number[]>; g?: Record<string, number[]> }

export default class BrainiacPlugin extends Plugin {
	settings: BrainSettings = { ...DEFAULT_SETTINGS };
	layoutCache: LayoutCache = { engine: "", u: {}, g: {} };

	async onload() {
		const data = ((await this.loadData()) ?? {}) as { settings?: Partial<BrainSettings>; layoutCache?: LayoutCache };
		this.settings = { ...DEFAULT_SETTINGS, ...(data.settings ?? {}) };
		this.layoutCache = data.layoutCache ?? { engine: "", u: {}, g: {} };
		this.registerView(VIEW_TYPE, leaf => new BrainView(leaf, this));
		this.addRibbonIcon("brain", "Open brain view", () => { void this.activate(); });
		this.addCommand({ id: "open", name: "Open brain view", callback: () => { void this.activate(); } });
	}

	async activate() {
		const ws = this.app.workspace;
		let leaf = ws.getLeavesOfType(VIEW_TYPE)[0];
		if (!leaf) {
			leaf = ws.getLeaf("tab");
			await leaf.setViewState({ type: VIEW_TYPE, active: true });
		}
		await ws.revealLeaf(leaf);
	}

	async saveSettings() {
		await this.saveData({ settings: this.settings, layoutCache: this.layoutCache });
	}
}
