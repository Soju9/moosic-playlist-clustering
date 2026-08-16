# Project Summary

## Objective

Create playlists with unsupervised machine learning, listen to the resulting clusters, and evaluate whether the song groups make sense musically.

## Why This Matters

Playlist generation is not only a clustering problem. Audio features can group songs by measurable similarities, but real user satisfaction depends on context, taste, mood, and behavior.

## What Was Built

- A starter K-Means demo on a small 10-song dataset.
- A larger clustering workflow on a 5,000-song dataset.
- Scaling experiments across multiple preprocessing methods.
- K-Means cluster evaluation using inertia and silhouette score.
- Playlist candidate generation from cluster labels.
- Qualitative evaluation of whether generated playlists sound coherent.

## Main Limitation

The model only uses track-level audio features. It does not know what users like, skip, save, replay, or listen to together.

## Data That Would Improve Results

- listening history
- skip rate
- replay rate
- likes and saves
- playlist adds/removals
- listening session context
- time of day
- user mood or activity
- genre and editorial tags
- collaborative filtering signals

## Portfolio Framing

This project is useful for employers because it shows technical ML skills and practical product thinking: the result is not treated as perfect just because a model produced clusters.

