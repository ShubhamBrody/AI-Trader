import { Routes, Route, Navigate } from 'react-router-dom';
import { Shell } from './components/layout/Shell';
import { DashboardPage } from '@/pages/DashboardPage';
import { MarketPage } from '@/pages/MarketPage';
import { CandlesPage } from '@/pages/CandlesPage';
import { IntradayPage } from '@/pages/IntradayPage';
import { StrategyPage } from '@/pages/StrategyPage';
import { AIPredictPage } from '@/pages/AIPredictPage';
import { RecommendationsPage } from '@/pages/RecommendationsPage';
import { BacktestPage } from '@/pages/BacktestPage';
import { NewsPage } from '@/pages/NewsPage';
import { PortfolioPage } from '@/pages/PortfolioPage';
import { PaperTradingPage } from '@/pages/PaperTradingPage';
import { JournalPage } from '@/pages/JournalPage';
import { HftIndexOptionsPage } from '@/pages/HftIndexOptionsPage';

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/market" element={<MarketPage />} />
        <Route path="/candles" element={<CandlesPage />} />
        <Route path="/intraday" element={<IntradayPage />} />
        <Route path="/strategy" element={<StrategyPage />} />
        <Route path="/ai" element={<AIPredictPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="/backtest" element={<BacktestPage />} />
        <Route path="/news" element={<NewsPage />} />
        <Route path="/portfolio" element={<PortfolioPage />} />
        <Route path="/paper" element={<PaperTradingPage />} />
        <Route path="/journal" element={<JournalPage />} />
        <Route path="/hft" element={<HftIndexOptionsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
