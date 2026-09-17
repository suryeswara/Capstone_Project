---
name: medverify-frontend
description: Reference for MedVerify AI's React/TypeScript/Vite frontend — page structure, component organization, state management, API integration, and how to add new pages or features.
---

# MedVerify AI — Frontend Reference

## Tech Stack

| Tool | Version/Config |
|------|---------------|
| Framework | React 18 + TypeScript |
| Build Tool | Vite |
| Styling | Tailwind CSS (`tailwind.config.js`) |
| Routing | React Router v6 |
| State | Zustand (in `src/store/`) |
| HTTP | Fetch API / custom hooks |
| Container | Nginx (serves `dist/` in Docker) |

---

## Pages (`src/pages/`)

| File | Route | Purpose |
|------|-------|---------|
| `Landing.tsx` | `/` | Marketing landing page (largest, 14KB) |
| `Verify.tsx` | `/verify` | Claim submission form + SSE progress display |
| `Dashboard.tsx` | `/dashboard` | User stats overview |
| `ReportPage.tsx` | `/report/:id` | Full verification report display |
| `HistoryPage.tsx` | `/history` | Past verifications list |
| `Consensus.tsx` | `/consensus` | Consensus score explorer |
| `EvidenceExplorer.tsx` | `/evidence` | Evidence browser |
| `FaithfulnessLab.tsx` | `/faithfulness` | Faithfulness score inspector |
| `Analytics.tsx` | `/analytics` | Usage analytics (admin) |
| `Admin.tsx` | `/admin` | Admin panel |
| `NotFound.tsx` | `*` | 404 page |

---

## State Management (`src/store/`)

Uses **Zustand** stores. Each store is a separate file.

### Pattern
```typescript
// src/store/useAuthStore.ts
import { create } from 'zustand'

interface AuthState {
  user: User | null
  token: string | null
  login: (email: string, password: string) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  login: async (email, password) => { /* ... */ },
  logout: () => set({ user: null, token: null }),
}))
```

---

## API Integration

**Base URL**: `http://localhost:8000` (dev) / `http://backend:8000` (Docker internal)

### Auth Headers
```typescript
const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${token}`
}
```

### SSE Streaming (Verify page)
```typescript
const eventSource = new EventSource(`/api/verifications/${id}/stream`)
eventSource.onmessage = (e) => {
  const data = JSON.parse(e.data)
  // data.status, data.progress, data.label
  setProgress(data.progress)
  if (data.status === 'COMPLETED') {
    eventSource.close()
    fetchReport(id)
  }
}
```

---

## Adding a New Page

1. Create `src/pages/MyPage.tsx`
2. Add route in `src/App.tsx`:
   ```tsx
   <Route path="/my-page" element={<MyPage />} />
   ```
3. Add nav link in the navigation component if needed
4. Add TypeScript types to `src/types/` if introducing new data shapes

---

## Building for Production

```bash
cd medverify-ai-frontend
npm run build
# Output: dist/ (served by Nginx in Docker)
```

### Nginx Config (`nginx.conf`)
- Serves `dist/` on port 80
- Proxies `/api` → `backend:8000`
- `try_files $uri /index.html` for SPA routing

---

## Docker Frontend Build

```dockerfile
# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

---

## Tailwind Config (`tailwind.config.js`)

Custom theme with MedVerify brand colors, typography, and component variants.  
Always use Tailwind classes; avoid inline styles unless absolutely necessary.  
Custom components (e.g. `.btn-primary`, `.card`) are defined in `src/index.css` via `@layer components`.
