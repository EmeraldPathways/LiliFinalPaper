"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { SummaryCount } from "@/lib/api";

type ChartCardProps = {
  title: string;
  data: SummaryCount[];
  color?: string;
};

export function ChartCard({ title, data, color = "#1e6b56" }: ChartCardProps) {
  const maxValue = data.reduce((max, item) => Math.max(max, item.value), 0);
  const isRatio = maxValue <= 1;

  return (
    <section className="card">
      <strong>{title}</strong>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(61,47,28,0.15)" />
            <XAxis dataKey="label" tick={{ fontSize: 12 }} />
            <YAxis
              allowDecimals={isRatio}
              domain={isRatio ? [0, 1] : undefined}
              tick={{ fontSize: 12 }}
            />
            <Tooltip />
            <Bar dataKey="value" fill={color} radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
