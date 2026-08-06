import { Routes, Route } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import Landing from '@/pages/Landing'
import Dashboard from '@/pages/Dashboard'
import Verify from '@/pages/Verify'
import EvidenceExplorer from '@/pages/EvidenceExplorer'
import Consensus from '@/pages/Consensus'
import FaithfulnessLab from '@/pages/FaithfulnessLab'
import HistoryPage from '@/pages/HistoryPage'
import Analytics from '@/pages/Analytics'
import Admin from '@/pages/Admin'
import ReportPage from '@/pages/ReportPage'
import NotFound from '@/pages/NotFound'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/app" element={<AppShell />}>
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="verify" element={<Verify />} />
        <Route path="evidence" element={<EvidenceExplorer />} />
        <Route path="consensus" element={<Consensus />} />
        <Route path="faithfulness" element={<FaithfulnessLab />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="admin" element={<Admin />} />
        <Route path="report/:id" element={<ReportPage />} />
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
