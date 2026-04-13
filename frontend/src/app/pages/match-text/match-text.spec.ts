import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MatchText } from './match-text';

describe('MatchText', () => {
  let component: MatchText;
  let fixture: ComponentFixture<MatchText>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MatchText]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MatchText);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
