---
title: Frontend Project Solution - Analytics Dashboard
sidebar_position: 2
description: In-depth frontend project walkthrough for a dashboard application.
---

# Frontend Project Solution: Analytics Dashboard

This is a strong frontend project because it combines layout, reusable components, API state, charting, accessibility, and performance.

## Problem statement

Build a dashboard that displays:

- summary cards
- charts
- filters
- tables
- loading and error states

## Main screens

- overview page
- campaign or project details
- filter drawer
- recent activity table

## Component structure

```text
src/
  components/
    SummaryCard.tsx
    ChartPanel.tsx
    FiltersBar.tsx
    DataTable.tsx
  pages/
    Dashboard.tsx
  hooks/
    useDashboardData.ts
```

## State flow

- filters in parent state
- data fetched based on filters
- reusable visual components receive already-prepared props

## Example React code

```tsx
import { useEffect, useState } from "react";

export function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/dashboard");
        const json = await res.json();
        setData(json);
      } catch {
        setError("Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>{error}</p>;

  return <section>{JSON.stringify(data)}</section>;
}
```

## UX details that make it stronger

- skeleton loaders
- empty states
- keyboard-accessible filters
- responsive layout
- color contrast and labels on charts

## Performance ideas

- paginate large tables
- defer heavy chart rendering
- memoize only where profiling justifies it
- cache network results

## Interview talking points

- why you split presentation and data-fetching components
- how loading and error states were handled
- what you did for accessibility
