import { create } from "zustand";
import { api } from "@/lib/api";
import type { FieldBBox, FieldPatchItem, FieldsResponse, GroundedField } from "@/lib/types";
import { clamp } from "@/lib/utils";

export interface PageMeta {
  page_number: number;
  width_px: number;
  height_px: number;
}

interface Snapshot {
  fieldsById: Record<string, GroundedField>;
  values: Record<string, string>;
}

interface EditorState {
  jobId: string | null;
  loading: boolean;
  loadError: string | null;

  pagesMeta: PageMeta[];
  fieldOrderByPage: Record<number, string[]>;
  fieldsById: Record<string, GroundedField>;
  fieldPageById: Record<string, number>;
  values: Record<string, string>;

  savedFieldsById: Record<string, GroundedField>;
  savedValues: Record<string, string>;

  currentPage: number;
  selectedFieldId: string | null;
  zoomPct: number;
  fitToPage: boolean;

  history: Snapshot[];
  future: Snapshot[];

  dirty: boolean;
  saving: boolean;
  refreshing: boolean;
  exporting: boolean;
  saveError: string | null;

  previewRunId: string | null;
  jobStatus: string | null;

  loadJob: (jobId: string) => Promise<void>;
  setCurrentPage: (page: number) => void;
  selectField: (fieldId: string | null) => void;
  setZoomPct: (pct: number) => void;
  setFitToPage: (fit: boolean) => void;

  beginDrag: (fieldId: string) => void;
  setBBoxLive: (fieldId: string, bbox: FieldBBox) => void;
  nudgeField: (fieldId: string, dx: number, dy: number) => void;
  updateValue: (fieldId: string, value: string) => void;
  updateFontSize: (fieldId: string, size: number) => void;
  toggleReviewed: (fieldId: string, reviewed: boolean) => void;
  resetPosition: (fieldId: string) => void;

  undo: () => void;
  redo: () => void;

  save: () => Promise<void>;
  refreshPreview: () => Promise<void>;
  exportPdf: () => Promise<{ download_url: string }>;
}

function snapshot(state: EditorState): Snapshot {
  return {
    fieldsById: { ...state.fieldsById },
    values: { ...state.values },
  };
}

function applyResponse(res: FieldsResponse) {
  const pagesMeta: PageMeta[] = [];
  const fieldOrderByPage: Record<number, string[]> = {};
  const fieldsById: Record<string, GroundedField> = {};
  const fieldPageById: Record<string, number> = {};

  for (const page of res.pages) {
    pagesMeta.push({ page_number: page.page_number, width_px: page.width_px, height_px: page.height_px });
    fieldOrderByPage[page.page_number] = page.fields.map((f) => f.field_id);
    for (const field of page.fields) {
      fieldsById[field.field_id] = field;
      fieldPageById[field.field_id] = page.page_number;
    }
  }

  return { pagesMeta, fieldOrderByPage, fieldsById, fieldPageById, values: { ...res.values } };
}

const MAX_HISTORY = 50;

export const useEditorStore = create<EditorState>((set, get) => ({
  jobId: null,
  loading: false,
  loadError: null,

  pagesMeta: [],
  fieldOrderByPage: {},
  fieldsById: {},
  fieldPageById: {},
  values: {},

  savedFieldsById: {},
  savedValues: {},

  currentPage: 1,
  selectedFieldId: null,
  zoomPct: 100,
  fitToPage: true,

  history: [],
  future: [],

  dirty: false,
  saving: false,
  refreshing: false,
  exporting: false,
  saveError: null,

  previewRunId: null,
  jobStatus: null,

  async loadJob(jobId: string) {
    set({ loading: true, loadError: null, jobId });
    try {
      const res = await api.getFields(jobId);
      const applied = applyResponse(res);
      set({
        ...applied,
        savedFieldsById: { ...applied.fieldsById },
        savedValues: { ...applied.values },
        currentPage: applied.pagesMeta[0]?.page_number ?? 1,
        selectedFieldId: null,
        history: [],
        future: [],
        dirty: false,
        loading: false,
        previewRunId: null,
      });
    } catch (err) {
      set({ loading: false, loadError: err instanceof Error ? err.message : "Failed to load fields" });
    }
  },

  setCurrentPage(page: number) {
    const max = get().pagesMeta.length;
    set({ currentPage: clamp(page, 1, Math.max(max, 1)), selectedFieldId: null });
  },

  selectField(fieldId: string | null) {
    set({ selectedFieldId: fieldId });
  },

  setZoomPct(pct: number) {
    set({ zoomPct: clamp(pct, 25, 300), fitToPage: false });
  },

  setFitToPage(fit: boolean) {
    set({ fitToPage: fit });
  },

  beginDrag(fieldId: string) {
    const state = get();
    if (!state.fieldsById[fieldId]) return;
    const before = snapshot(state);
    set({ history: [...state.history.slice(-MAX_HISTORY + 1), before], future: [] });
  },

  setBBoxLive(fieldId: string, bbox: FieldBBox) {
    const state = get();
    const field = state.fieldsById[fieldId];
    if (!field) return;
    const pageMeta = state.pagesMeta.find((p) => p.page_number === state.fieldPageById[fieldId]);
    const maxX = pageMeta ? pageMeta.width_px - field.bbox.w : Number.MAX_SAFE_INTEGER;
    const maxY = pageMeta ? pageMeta.height_px - field.bbox.h : Number.MAX_SAFE_INTEGER;
    const nextBbox: FieldBBox = {
      ...bbox,
      x: Math.round(clamp(bbox.x, 0, Math.max(maxX, 0))),
      y: Math.round(clamp(bbox.y, 0, Math.max(maxY, 0))),
    };
    set({ fieldsById: { ...state.fieldsById, [fieldId]: { ...field, bbox: nextBbox } }, dirty: true });
  },

  nudgeField(fieldId: string, dx: number, dy: number) {
    const state = get();
    const field = state.fieldsById[fieldId];
    if (!field) return;
    const pageMeta = state.pagesMeta.find((p) => p.page_number === state.fieldPageById[fieldId]);
    const maxX = pageMeta ? pageMeta.width_px - field.bbox.w : Number.MAX_SAFE_INTEGER;
    const maxY = pageMeta ? pageMeta.height_px - field.bbox.h : Number.MAX_SAFE_INTEGER;
    const nextBbox: FieldBBox = {
      ...field.bbox,
      x: Math.round(clamp(field.bbox.x + dx, 0, Math.max(maxX, 0))),
      y: Math.round(clamp(field.bbox.y + dy, 0, Math.max(maxY, 0))),
    };
    const before = snapshot(state);
    set({
      fieldsById: { ...state.fieldsById, [fieldId]: { ...field, bbox: nextBbox } },
      history: [...state.history.slice(-MAX_HISTORY + 1), before],
      future: [],
      dirty: true,
    });
  },

  updateValue(fieldId: string, value: string) {
    const state = get();
    if (state.values[fieldId] === value) return;
    const before = snapshot(state);
    set({
      values: { ...state.values, [fieldId]: value },
      history: [...state.history.slice(-MAX_HISTORY + 1), before],
      future: [],
      dirty: true,
    });
  },

  updateFontSize(fieldId: string, size: number) {
    const state = get();
    const field = state.fieldsById[fieldId];
    if (!field) return;
    const clamped = clamp(size, 4, 96);
    if (field.font_size_pt === clamped) return;
    const before = snapshot(state);
    set({
      fieldsById: { ...state.fieldsById, [fieldId]: { ...field, font_size_pt: clamped } },
      history: [...state.history.slice(-MAX_HISTORY + 1), before],
      future: [],
      dirty: true,
    });
  },

  toggleReviewed(fieldId: string, reviewed: boolean) {
    const state = get();
    const field = state.fieldsById[fieldId];
    if (!field) return;
    const before = snapshot(state);
    set({
      fieldsById: { ...state.fieldsById, [fieldId]: { ...field, reviewed } },
      history: [...state.history.slice(-MAX_HISTORY + 1), before],
      future: [],
      dirty: true,
    });
  },

  resetPosition(fieldId: string) {
    const state = get();
    const original = state.savedFieldsById[fieldId];
    const field = state.fieldsById[fieldId];
    if (!original || !field) return;
    const before = snapshot(state);
    set({
      fieldsById: { ...state.fieldsById, [fieldId]: { ...field, bbox: { ...original.bbox } } },
      history: [...state.history.slice(-MAX_HISTORY + 1), before],
      future: [],
      dirty: true,
    });
  },

  undo() {
    const state = get();
    const prev = state.history[state.history.length - 1];
    if (!prev) return;
    const currentSnap = snapshot(state);
    set({
      fieldsById: prev.fieldsById,
      values: prev.values,
      history: state.history.slice(0, -1),
      future: [...state.future, currentSnap],
      dirty: state.history.length > 1,
    });
  },

  redo() {
    const state = get();
    const next = state.future[state.future.length - 1];
    if (!next) return;
    const currentSnap = snapshot(state);
    set({
      fieldsById: next.fieldsById,
      values: next.values,
      future: state.future.slice(0, -1),
      history: [...state.history, currentSnap],
      dirty: true,
    });
  },

  async save() {
    const state = get();
    if (!state.jobId) return;
    set({ saving: true, saveError: null });
    try {
      const patchItems: FieldPatchItem[] = Object.values(state.fieldsById).map((field) => ({
        field_id: field.field_id,
        page_number: state.fieldPageById[field.field_id],
        bbox: field.bbox,
        font_size_pt: field.font_size_pt ?? undefined,
        reviewed: field.reviewed ?? false,
      }));
      await api.patchFields(state.jobId, patchItems);
      await api.patchValues(state.jobId, state.values);
      set({
        savedFieldsById: { ...state.fieldsById },
        savedValues: { ...state.values },
        dirty: false,
        saving: false,
        history: [],
        future: [],
      });
    } catch (err) {
      set({ saving: false, saveError: err instanceof Error ? err.message : "Save failed" });
      throw err;
    }
  },

  async refreshPreview() {
    const state = get();
    if (!state.jobId) return;
    set({ refreshing: true, saveError: null });
    try {
      if (state.dirty) {
        await state.save();
      }
      const res = await api.stampImages(state.jobId);
      set({ previewRunId: res.run_id, refreshing: false });
    } catch (err) {
      set({ refreshing: false, saveError: err instanceof Error ? err.message : "Refresh preview failed" });
      throw err;
    }
  },

  async exportPdf() {
    const state = get();
    if (!state.jobId) throw new Error("No job loaded");
    set({ exporting: true, saveError: null });
    try {
      if (state.dirty) {
        await state.save();
      }
      const res = await api.stampPdf(state.jobId);
      set({ exporting: false });
      return res;
    } catch (err) {
      set({ exporting: false, saveError: err instanceof Error ? err.message : "Export failed" });
      throw err;
    }
  },
}));
