// Shapes mirror harness/specs/api-contract.md and harness/specs/field-schema.md.

export type JobStatus = "converting" | "grounding" | "ready" | "failed" | "exported";

export interface JobListItem {
  job_id: string;
  source_filename: string;
  page_count: number;
  status: JobStatus;
  created_at: string;
}

export interface JobListResponse {
  jobs: JobListItem[];
}

export interface JobStageError {
  stage?: string;
  message?: string;
  page_index?: number;
  status?: string;
  error?: string;
  detail?: string;
  [key: string]: unknown;
}

export interface JobGroundingStage {
  grounded_pages?: number;
  total_pages?: number;
  status?: string;
}

export interface JobDetail {
  job_id: string;
  source_filename: string | null;
  status: JobStatus;
  page_count: number;
  stages: {
    grounding?: JobGroundingStage;
    [key: string]: unknown;
  };
  errors: JobStageError[];
  artifacts: {
    has_stamped_preview: boolean;
    has_export_pdf: boolean;
  };
}

export type FieldType = "text" | "multiline_text" | "checkbox" | "radio";

export type GroundingSource = "cell" | "line_anchor" | "label_anchor" | "pixel" | "template" | null;

export type QaStatus = "confirmed" | "adjusted" | "flagged" | null;

export interface FieldBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface GroundedField {
  field_id: string;
  type: FieldType;
  bbox: FieldBBox;
  confidence: number;
  label: string;
  evidence?: { line_ids?: string[] };
  grounding_type?: string;
  grounding_source?: GroundingSource;
  qa_status?: QaStatus;
  reviewed?: boolean;
  font_size_pt?: number | null;
}

export interface FieldsPage {
  page_number: number;
  width_px: number;
  height_px: number;
  fields: GroundedField[];
}

export interface FieldsResponse {
  job_id: string;
  style: { font_size_pt?: number; text_color?: string; [key: string]: unknown };
  pages: FieldsPage[];
  values: Record<string, string>;
}

export interface FieldPatchItem {
  field_id: string;
  page_number: number;
  bbox?: FieldBBox;
  font_size_pt?: number;
  reviewed?: boolean;
}

export interface PatchFieldsResponse {
  fields: GroundedField[];
}

export interface PatchValuesResponse {
  values: Record<string, string>;
  style: Record<string, unknown>;
}

export interface StampImagesRunResponse {
  run_id: string;
  pages: { page_number: number; image_url: string }[];
}

export interface StampPdfRunResponse {
  run_id: string;
  download_url: string;
}

export interface ApiErrorBody {
  error: string;
  message: string;
}
