import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  {
    path: 'home',
    loadComponent: () => import('./pages/home/home').then(m => m.HomeComponent)
  },
  {
    path: 'match-text',
    loadComponent: () => import('./pages/match-text/match-text').then(m => m.MatchTextComponent)
  },
  {
    path: 'match-file',
    loadComponent: () => import('./pages/match-file/match-file').then(m => m.MatchFileComponent)
  },
  {
    path: 'jobs',
    loadComponent: () => import('./pages/jobs/jobs').then(m => m.JobsComponent)
  },
  {
    path: 'stats',
    loadComponent: () => import('./pages/stats/stats').then(m => m.StatsComponent)
  },
  { path: '**', redirectTo: 'home' }
];