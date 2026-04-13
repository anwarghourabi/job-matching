export interface MatchRequest {
  name: string;
  cv_text: string;
  experience_level: string;
  desired_location: string;
  min_salary: number;
  max_salary: number;
  remote_only: boolean;
  employment_type: string;
  top_k: number;
}

export interface JobResult {
  rank: number;
  job_title: string;
  location: string;
  salary_usd: number;
  has_salary: boolean;
  experience_level: string;
  employment_type: string;
  remote_ratio: number;
  company_size: string;
  source: string;
  sbert_score: number;
  filter_bonus: number;
  final_score: number;
  match_reasons: string[];
}

export interface MatchResponse {
  candidate_name: string;
  experience_level: string;
  domain_detected: string;
  skills_detected: string[];
  total_jobs_scanned: number;
  eligible_jobs: number;
  results: JobResult[];
  top_score: number;
  avg_score: number;
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  engine_ready: boolean;
  corpus_size: number;
  model_name: string;
  version: string;
}

export interface StatsResponse {
  total_jobs: number;
  jobs_with_salary: number;
  jobs_remote: number;
  unique_locations: number;
  unique_titles: number;
  sources: Record<string, number>;
  experience_distribution: Record<string, number>;
  employment_type_distribution: Record<string, number>;
}

export interface JobsResponse {
  total: number;
  page: number;
  per_page: number;
  pages: number;
  jobs: any[];
}