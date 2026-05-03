import { Component, Input, Output, EventEmitter, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api';
import { AuthService } from '../../core/services/auth';

@Component({
  selector: 'app-job-crud-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './job-crud-modal.html',
  styleUrl: './job-crud-modal.css',
})
export class JobCrudModal implements OnInit {
  @Input() mode: 'create' | 'edit' = 'create';
  @Input() job: any = null;
  @Output() closed = new EventEmitter<boolean>();

  private apiService = inject(ApiService);
  private auth = inject(AuthService);

  customJobs: any[] = [];
  loading = false;
  message = '';

  form = {
    job_title: '',
    description: '',
    skills_desc: '',
    experience_level: 'mid',
    location: '',
    salary_usd: 0,
    remote_ratio: 0
  };

  ngOnInit() {
    this.loadCustomJobs();
    if (this.mode === 'edit' && this.job) {
      this.form = { ...this.job };
    }
  }

  loadCustomJobs() {
    this.apiService.getCustomJobs().subscribe(jobs => this.customJobs = jobs);
  }

  submit() {
    this.loading = true;
    const obs = this.mode === 'create'
      ? this.apiService.createJob(this.form)
      : this.apiService.updateJob(this.job.id, this.form);

    obs.subscribe({
      next: (res) => {
        this.message = res.message;
        this.loading = false;
        this.loadCustomJobs();
        if (this.mode === 'create') this.resetForm();
      },
      error: () => {
        this.message = '❌ Erreur lors de la sauvegarde';
        this.loading = false;
      }
    });
  }

  deleteJob(id: number) {
    if (!confirm('Supprimer cette offre ?')) return;
    this.apiService.deleteJob(id).subscribe({
      next: () => this.loadCustomJobs(),
      error: () => this.message = '❌ Erreur suppression'
    });
  }

  editJob(job: any) {
    this.mode = 'edit';
    this.job = job;
    this.form = {
      job_title:        job.job_title        || '',
      description:      job.description      || '',
      skills_desc:      job.skills_desc      || '',
      experience_level: job.experience_level || 'mid',
      location:         job.location         || '',
      salary_usd:       job.salary_usd       || 0,
      remote_ratio:     job.remote_ratio     || 0,
    };
  }

  resetForm() {
    this.mode = 'create';
    this.job = null;
    this.form = {
      job_title: '', description: '', skills_desc: '',
      experience_level: 'mid', location: '', salary_usd: 0, remote_ratio: 0
    };
  }

  get isLoggedIn() { return this.auth.isLoggedIn(); }

  close() { this.closed.emit(true); }
}