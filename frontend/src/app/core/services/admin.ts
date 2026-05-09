import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from './auth';

@Injectable({ providedIn: 'root' })
export class AdminService {
  private http = inject(HttpClient);
  private auth = inject(AuthService);
  private BASE = 'http://localhost:8000/admin';

  private get headers() {
    return { Authorization: `Bearer ${this.auth.token()}` };
  }

  // ── Dashboard ──────────────────────────────────────────────
  getDashboard(): Observable<any> {
    return this.http.get(`${this.BASE}/dashboard`, { headers: this.headers });
  }

  // ── Users ──────────────────────────────────────────────────
  getUsers(search = '', role = '', page = 1, limit = 20): Observable<any> {
    let params = new HttpParams()
      .set('page', page).set('limit', limit);
    if (search) params = params.set('search', search);
    if (role)   params = params.set('role', role);
    return this.http.get(`${this.BASE}/users`, { headers: this.headers, params });
  }

  updateUserRole(userId: number, role: string): Observable<any> {
    return this.http.put(
      `${this.BASE}/users/${userId}/role`,
      { role },
      { headers: this.headers }
    );
  }

  deleteUser(userId: number): Observable<any> {
    return this.http.delete(`${this.BASE}/users/${userId}`, { headers: this.headers });
  }

  // ── Jobs ───────────────────────────────────────────────────
  getJobs(search = '', page = 1, limit = 20): Observable<any> {
    let params = new HttpParams().set('page', page).set('limit', limit);
    if (search) params = params.set('search', search);
    return this.http.get(`${this.BASE}/jobs`, { headers: this.headers, params });
  }

  deleteJob(jobId: number): Observable<any> {
    return this.http.delete(`${this.BASE}/jobs/${jobId}`, { headers: this.headers });
  }

  // ── Applications ───────────────────────────────────────────
  getApplications(status = '', page = 1, limit = 20): Observable<any> {
    let params = new HttpParams().set('page', page).set('limit', limit);
    if (status) params = params.set('status', status);
    return this.http.get(`${this.BASE}/applications`, { headers: this.headers, params });
  }
}