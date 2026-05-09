import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

export interface User {
  id: number;
  email: string;
  full_name: string;
  role?: string;
  cv_text?: string;
  experience_level?: string;
  desired_location?: string;
  skills?: string;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private BASE = 'http://localhost:8000';

  currentUser = signal<User | null>(null);
  token       = signal<string | null>(null);

  constructor() {
    const savedToken = localStorage.getItem('token');
    const savedUser  = localStorage.getItem('user');

    if (savedToken) {
      this.token.set(savedToken);

      // Restaurer immédiatement depuis localStorage (évite le flash au refresh)
      if (savedUser) {
        try { this.currentUser.set(JSON.parse(savedUser)); } catch { }
      }

      this.loadMe(); // sync en arrière-plan
    }
  }

  signup(email: string, password: string, full_name: string): Observable<any> {
    return this.http.post(`${this.BASE}/auth/signup`, { email, password, full_name }).pipe(
      tap((res: any) => this.saveSession(res))
    );
  }

  sendVerificationCode(
    email: string, password: string,
    confirm_password: string, full_name: string,
    role: 'candidate' | 'recruiter'
  ): Observable<any> {
    const roleMap: Record<string, string> = {
      'candidate': 'candidat',
      'recruiter': 'recruteur'
    };
    return this.http.post(`${this.BASE}/auth/send-code`, {
      email, password, confirm_password, full_name, role: roleMap[role]
    });
  }

  verifyCode(email: string, code: string): Observable<any> {
    return this.http.post(`${this.BASE}/auth/verify-code`, { email, code }).pipe(
      tap((res: any) => this.saveSession(res))
    );
  }

  login(email: string, password: string): Observable<any> {
    return this.http.post(`${this.BASE}/auth/login`, { email, password }).pipe(
      tap((res: any) => this.saveSession(res))
    );
  }

  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    this.token.set(null);
    this.currentUser.set(null);
  }

  loadMe() {
    this.http.get<User>(`${this.BASE}/auth/me`, { headers: this.authHeaders() }).subscribe({
      next: user => {
        this.currentUser.set(user);
        localStorage.setItem('user', JSON.stringify(user)); // sync localStorage
      },
      error: () => this.logout()
    });
  }

  updateProfile(data: Partial<User>): Observable<any> {
    return this.http.put(`${this.BASE}/auth/profile`, data, { headers: this.authHeaders() }).pipe(
      tap(() => this.loadMe())
    );
  }

  addToHistory(job: { job_title: string; job_source: string; score: number }): Observable<any> {
    return this.http.post(`${this.BASE}/auth/history`, job, { headers: this.authHeaders() });
  }

  getHistory(): Observable<any[]> {
    return this.http.get<any[]>(`${this.BASE}/auth/history`, { headers: this.authHeaders() });
  }

  getRecommendations(): Observable<any> {
    return this.http.get<any>(`${this.BASE}/recommendations`, { headers: this.authHeaders() });
  }

  applyToJob(data: {
    job_title: string; job_source: string;
    job_location: string; salary_usd: number; cover_letter: string;
  }): Observable<any> {
    return this.http.post(`${this.BASE}/auth/apply`, data, { headers: this.authHeaders() });
  }

  getApplications(): Observable<any[]> {
    return this.http.get<any[]>(`${this.BASE}/auth/applications`, { headers: this.authHeaders() });
  }

  getRecruiterApplications(): Observable<any[]> {
    return this.http.get<any[]>(`${this.BASE}/auth/recruiter/applications`, { headers: this.authHeaders() });
  }

  updateApplicationStatus(appId: number, status: 'acceptée' | 'rejetée'): Observable<any> {
    return this.http.put(
      `${this.BASE}/auth/recruiter/applications/${appId}`,
      { status },
      { headers: this.authHeaders() }
    );
  }

  uploadCv(file: File): Observable<any> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post(
      `${this.BASE}/auth/profile/cv`,
      form,
      { headers: { Authorization: `Bearer ${this.token()}` } }
    );
  }

  isLoggedIn(): boolean {
    return !!this.token();
  }

  authHeaders() {
    return { Authorization: `Bearer ${this.token()}` };
  }

  private saveSession(res: any) {
    localStorage.setItem('token', res.token);
    localStorage.setItem('user', JSON.stringify(res.user));
    this.token.set(res.token);
    this.currentUser.set(res.user);
  }
}