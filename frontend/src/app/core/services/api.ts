import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  MatchRequest, MatchResponse, HealthResponse,
  StatsResponse, JobsResponse
} from '../models/api.models';
export interface CustomJob {
  id?: number;
  job_title: string;
  description?: string;
  skills_desc?: string;
  experience_level?: string;
  location?: string;
  salary_usd?: number;
  remote_ratio?: number;
  created_at?: string;
}
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly BASE = 'http://localhost:8000';

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.BASE}/health`);
  }

  stats(): Observable<StatsResponse> {
    return this.http.get<StatsResponse>(`${this.BASE}/stats`);
  }

  matchText(req: MatchRequest): Observable<MatchResponse> {
    return this.http.post<MatchResponse>(`${this.BASE}/match/text`, req);
  }

  matchFile(
    file: File,
    params: {
      experience_level?: string;
      desired_location?: string;
      min_salary?: number;
      remote_only?: boolean;
      employment_type?: string;
      top_k?: number;
    }
  ): Observable<MatchResponse> {
    const formData = new FormData();
    formData.append('file', file);

    let httpParams = new HttpParams();
    if (params.experience_level) httpParams = httpParams.set('experience_level', params.experience_level);
    if (params.desired_location) httpParams = httpParams.set('desired_location', params.desired_location);
    if (params.min_salary !== undefined) httpParams = httpParams.set('min_salary', params.min_salary.toString());
    if (params.remote_only !== undefined) httpParams = httpParams.set('remote_only', params.remote_only.toString());
    if (params.employment_type) httpParams = httpParams.set('employment_type', params.employment_type);
    if (params.top_k !== undefined) httpParams = httpParams.set('top_k', params.top_k.toString());

    return this.http.post<MatchResponse>(`${this.BASE}/match/file`, formData, { params: httpParams });
  }

  listJobs(page = 1, perPage = 20, source = '', level = ''): Observable<JobsResponse> {
    let params = new HttpParams()
      .set('page', page.toString())
      .set('per_page', perPage.toString());
    if (source) params = params.set('source', source);
    if (level) params = params.set('level', level);
    return this.http.get<JobsResponse>(`${this.BASE}/jobs`, { params });
  }

  searchJobs(q: string, top = 20): Observable<any> {
    const params = new HttpParams().set('q', q).set('top', top.toString());
    return this.http.get<any>(`${this.BASE}/jobs/search`, { params });
  }

    // ══════════════════════════════════════════════════════════
  // CRUD Offres personnalisées
  // ══════════════════════════════════════════════════════════

  getCustomJobs(): Observable<CustomJob[]> {
    return this.http.get<CustomJob[]>(`${this.BASE}/crud/jobs`);
  }

  createJob(job: CustomJob): Observable<{ id: number; message: string }> {
    return this.http.post<{ id: number; message: string }>(`${this.BASE}/crud/jobs`, job);
  }

  updateJob(id: number, job: Partial<CustomJob>): Observable<{ message: string }> {
    return this.http.put<{ message: string }>(`${this.BASE}/crud/jobs/${id}`, job);
  }

  deleteJob(id: number): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.BASE}/crud/jobs/${id}`);
  }
}