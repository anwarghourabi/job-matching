import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

export interface User {
  id: number;
  email: string;
  full_name: string;
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
  token = signal<string | null>(null);

  constructor() {
    const saved = localStorage.getItem('token');
    if (saved) {
      this.token.set(saved);
      this.loadMe();
    }
  }

  signup(email: string, password: string, full_name: string): Observable<any> {
    return this.http.post(`${this.BASE}/auth/signup`, { email, password, full_name }).pipe(
      tap((res: any) => this.saveSession(res))
    );
  }
  sendVerificationCode(email: string, password: string, confirm_password: string, full_name: string): Observable<any> {
    return this.http.post(`${this.BASE}/auth/send-code`, { email, password, confirm_password, full_name });
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
    this.token.set(null);
    this.currentUser.set(null);
  }

  loadMe() {
    this.http.get<User>(`${this.BASE}/auth/me`, { headers: this.authHeaders() }).subscribe({
      next: user => this.currentUser.set(user),
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

  isLoggedIn(): boolean {
    return !!this.token();
  }

  authHeaders() {
    return { Authorization: `Bearer ${this.token()}` };
  }

  private saveSession(res: any) {
    localStorage.setItem('token', res.token);
    this.token.set(res.token);
    this.currentUser.set(res.user);
  }
}