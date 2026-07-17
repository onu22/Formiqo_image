import type {
  ApiErrorBody,
  FieldPatchItem,
  FieldsResponse,
  JobDetail,
  JobListResponse,
  PatchFieldsResponse,
  PatchValuesResponse,
  StampImagesRunResponse,
  StampPdfRunResponse,
} from "./types";

const API_BASE = "/api/v1";

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(status: number, body: Partial<ApiErrorBody>) {
    super(body.message || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.code = body.error || "unknown_error";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    let body: Partial<ApiErrorBody> = {};
    try {
      body = await res.json();
    } catch {
      // ignore non-JSON error bodies
    }
    throw new ApiError(res.status, body);
  }

  if (res.status === 204) {
    return undefined as T;
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return undefined as T;
}

export function pageImageUrl(
  jobId: string,
  pageNumber: number,
  variant: "source" | "stamped" = "source",
  cacheBust?: string | number,
): string {
  const params = new URLSearchParams({ variant });
  if (cacheBust !== undefined) params.set("v", String(cacheBust));
  return `${API_BASE}/jobs/${jobId}/pages/${pageNumber}/image?${params.toString()}`;
}

export function exportDownloadUrl(jobId: string): string {
  return `${API_BASE}/jobs/${jobId}/export`;
}

export const api = {
  createJob(file: File): Promise<{ job_id: string; status: string }> {
    const formData = new FormData();
    formData.append("file", file);
    return request("/jobs", { method: "POST", body: formData });
  },

  listJobs(): Promise<JobListResponse> {
    return request("/jobs");
  },

  getJob(jobId: string): Promise<JobDetail> {
    return request(`/jobs/${jobId}`);
  },

  deleteJob(jobId: string): Promise<void> {
    return request(`/jobs/${jobId}`, { method: "DELETE" });
  },

  getFields(jobId: string): Promise<FieldsResponse> {
    return request(`/jobs/${jobId}/fields`);
  },

  patchFields(jobId: string, fields: FieldPatchItem[]): Promise<PatchFieldsResponse> {
    return request(`/jobs/${jobId}/fields`, {
      method: "PATCH",
      body: JSON.stringify({ fields }),
    });
  },

  patchValues(
    jobId: string,
    values: Record<string, string>,
  ): Promise<PatchValuesResponse> {
    return request(`/jobs/${jobId}/values`, {
      method: "PATCH",
      body: JSON.stringify({ values }),
    });
  },

  stampImages(jobId: string): Promise<StampImagesRunResponse> {
    return request(`/jobs/${jobId}/stamp-images`, { method: "POST" });
  },

  stampPdf(jobId: string): Promise<StampPdfRunResponse> {
    return request(`/jobs/${jobId}/stamp-pdf`, { method: "POST" });
  },

  refineGrounding(jobId: string): Promise<{ status: string }> {
    // E5 vision QA refinement re-run; poll getJob for stages.qa_refine afterwards.
    return request(`/jobs/${jobId}/refine-grounding`, { method: "POST" });
  },

  retryGrounding(jobId: string): Promise<unknown> {
    // Legacy job-scoped re-run; closest available mechanism until a dedicated
    // retry-failed-pages endpoint ships. Uses default provider/model on the job.
    return request(`/jobs/${jobId}/ground-fields-from-lines`, { method: "POST" });
  },
};
