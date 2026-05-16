import React, { useState } from 'react';

const MetricCard = ({ title, value, formatAsPercent = false }) => (
    <div className="bg-gray-100 p-4 rounded-lg text-center">
        <h4 className="text-sm font-medium text-gray-500">{title}</h4>
        <p className="text-xl font-semibold text-gray-800">
            {typeof value === 'number' ? (formatAsPercent ? `${(value * 100).toFixed(2)}%` : value.toFixed(2)) : 'N/A'}
        </p>
    </div>
);

const IterationTable = ({ iterations }) => (
    <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 mt-4">
            <thead className="bg-gray-50">
                <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Iteration</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Config</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CAGR</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Max Drawdown</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Win Rate</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Expectancy</th>
                </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
                {iterations.map((iter) => (
                    <tr key={iter.iteration}>
                        <td className="px-6 py-4 whitespace-nowrap font-medium">{iter.iteration}</td>
                        <td className="px-6 py-4 whitespace-nowrap font-mono text-sm">{JSON.stringify(iter.config)}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{(iter.metrics.cagr * 100).toFixed(2)}%</td>
                        <td className="px-6 py-4 whitespace-nowrap">{iter.metrics.max_drawdown.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{(iter.metrics.win_rate * 100).toFixed(2)}%</td>
                        <td className="px-6 py-4 whitespace-nowrap font-semibold">{iter.metrics.expectancy.toFixed(2)}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);


function App() {
  const [prompt, setPrompt] = useState("Trade AAPL. Buy when the 15-day EMA crosses above the 50-day EMA. Sell when it crosses below.");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-50 min-h-screen">
        <div className="container mx-auto p-4 sm:p-6 lg:p-8">
            <header className="text-center mb-8">
                <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">AI-Driven Algorithmic Trading Backtester</h1>
                <p className="text-md text-gray-600 mt-2">
                    Input a strategy, and let an AI agent optimize it for you.
                </p>
            </header>

            <div className="max-w-2xl mx-auto bg-white p-6 rounded-lg shadow-md">
                <form onSubmit={handleSubmit}>
                    <textarea
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    rows="4"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g., 'Trade AAPL. Buy when the 50-day EMA crosses above the 200-day EMA. Sell when it crosses below.'"
                    />
                    <button type="submit" className="mt-4 w-full px-4 py-2 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400" disabled={loading}>
                    {loading ? 'Optimizing Strategy...' : 'Run Backtest & Optimize'}
                    </button>
                </form>
            </div>

            {loading && <div className="text-center mt-8">Running... Please wait. This may take a minute.</div>}
            {error && <div className="mt-8 max-w-2xl mx-auto p-4 bg-red-100 text-red-800 border border-red-300 rounded-md">{`Error: ${error}`}</div>}

            {results && (
            <div className="mt-10">
                {/* Best Configuration Section */}
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-2xl font-bold text-center text-gray-800">Best Configuration Found</h2>
                    <div className="mt-4 p-6 bg-white rounded-lg shadow-lg border border-green-500">
                        <div className="text-center mb-4">
                            <span className="font-mono text-lg p-2 bg-gray-100 rounded-md">{JSON.stringify(results.best_configuration.config)}</span>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                            <MetricCard title="CAGR" value={results.best_configuration.metrics.cagr} formatAsPercent />
                            <MetricCard title="Max Drawdown" value={results.best_configuration.metrics.max_drawdown} />
                            <MetricCard title="Win Rate" value={results.best_configuration.metrics.win_rate} formatAsPercent />
                            <MetricCard title="Expectancy" value={results.best_configuration.metrics.expectancy} />
                            <MetricCard title="Avg. Win" value={results.best_configuration.metrics.avg_win} />
                            <MetricCard title="Avg. Loss" value={results.best_configuration.metrics.avg_loss} />
                        </div>
                    </div>
                </div>

                {/* Iteration Comparison Section */}
                <div className="max-w-6xl mx-auto mt-10">
                    <h2 className="text-2xl font-bold text-center text-gray-800 mb-4">Optimization Iterations</h2>
                    <IterationTable iterations={results.all_iterations} />
                </div>
            </div>
            )}
        </div>
    </div>
  );
}

export default App;
