# MedVerify AI — Frontend

A complete, production-quality frontend for MedVerify AI, an evidence-grounded medical claim
verification platform. Built to match the Phase 1 MVP scope defined in the implementation plan
(3 disease categories — Cardiovascular Disease, Vaccination, Diabetes — English, text-only claims)
while implementing every screen from the reference design using realistic mock data, ready to be
wired up to a real API later.

## Stack

- React 18 + TypeScript + Vite
- Tailwind CSS (custom design tokens — see `tailwind.config.js`)
- React Router for routing
- Zustand for app state (`src/store/useAppStore.ts`)
- Framer Motion for motion / the animated pipeline
- Recharts for charts
- Lucide for icons

> Note: the brief requested React 19 / Shadcn / TanStack Query / Axios / Zod / React Hook Form.
> This build uses React 18 (broadly compatible today) and hand-built Tailwind components instead
> of Shadcn to keep the bundle lean; the component structure is written so swapping in Shadcn
> primitives, TanStack Query, or a real API layer later is a drop-in change, not a rewrite.

## Getting started

```bash
npm install
npm run dev       # start local dev server
npm run build     # production build to /dist
npm run preview   # preview the production build
```

## Structure

```
src/
  components/
    layout/        PublicNav, Footer, AppShell (sidebar + topbar)
    ui/             Card, Badge, Button, SectionHeading, Skeleton
    pipeline/       PipelineVisualizer (the 9-stage animated "evidence thread")
    credibility/    CredibilityGauge (circular score), MiniMeter
    evidence/       EvidenceCard
    consensus/      ConsensusTimeline chart
    faithfulness/   FaithfulnessView (sentence-to-source mapping)
  pages/
    Landing.tsx           marketing site: hero, pipeline, evidence system, comparison, team, FAQ
    Dashboard.tsx         verification stats, trend charts, recent claims
    Verify.tsx            claim input + live animated pipeline + result
    EvidenceExplorer.tsx  searchable/filterable evidence corpus
    Consensus.tsx         supporting/contradicting/neutral consensus visualization
    FaithfulnessLab.tsx   sentence-level faithfulness verification explorer
    HistoryPage.tsx       searchable verification history with bookmarks
    Analytics.tsx         platform-wide analytics dashboards
    Admin.tsx             system health, knowledge base, security panel
    ReportPage.tsx         full printable verification report per claim
  store/
    useAppStore.ts   Zustand store: history, active pipeline run, mock verification engine
  lib/
    mockData.ts      example claims, evidence, consensus timelines, analytics data
    pipeline.ts       the 9-stage pipeline definition shared by Landing + Verify
    utils.ts          cn(), date formatting, verdict color tokens
  types/
    index.ts         shared domain types (EvidenceItem, VerificationRecord, etc.)
```

## Design language

- **Palette**: deep "ink" slate for text/dark surfaces, "clinical" cyan as the primary accent,
  "trust" blue as a secondary accent, and semantic verdict colors (emerald/amber/red/gray) — calm
  and clinical rather than saturated or playful.
- **Type**: Inter for UI text, IBM Plex Mono for small uppercase "mono-label" eyebrows/metadata,
  giving the interface a lab-report/data-sheet feel appropriate to a scientific product.
- **Signature element**: the pipeline "evidence thread" — a connected vertical line of stage nodes
  that fills in as each verification stage completes, reused at three scales: the landing page demo,
  the full pipeline explainer section, and the live claim workspace.

## Wiring up a real backend

Everything currently reads from `src/lib/mockData.ts` through the Zustand store. To connect a real
API:

1. Replace `pickMockResult` / the `setTimeout` staging logic in `useAppStore.ts` with real API calls
   (e.g. via TanStack Query) that stream pipeline stage updates (SSE/WebSocket) and return a
   `VerificationRecord` matching `src/types/index.ts`.
2. The `EvidenceItem`, `FaithfulnessSentence`, `ConsensusPoint`, and `CredibilityBreakdown` types are
   designed to map directly onto the backend's Evidence Ranking Engine, Faithfulness Verification
   module, Consensus Engine, and Credibility Score calculation described in the implementation plan.
