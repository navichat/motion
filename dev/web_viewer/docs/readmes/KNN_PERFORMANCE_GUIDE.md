# KNN Performance Measurement Guide

## What "Average Accuracy" Really Means

When we say a KNN algorithm has "85% accuracy," we mean that **85% of the neighbors it returns are actually in the true top-K closest vectors** as determined by exhaustive brute-force search.

## Key Metrics Explained

### 1. Precision@K
**Definition**: Of the K neighbors returned, what fraction are actually in the TRUE top-K closest vectors?

**Formula**: `(Number of correct neighbors found) / K`

**Example**: If algorithm returns 8 neighbors and 6 of them are actually in the true top-8, then Precision@8 = 6/8 = 75%

**Why it matters**: This tells you if the algorithm is finding the *right* neighbors, not just *some* neighbors.

### 2. NDCG@K (Normalized Discounted Cumulative Gain)
**Definition**: Measures ranking quality - did the algorithm get the ORDER of neighbors correct?

**Range**: 0.0 (terrible ranking) to 1.0 (perfect ranking)

**Why it matters**: Two algorithms might both have 80% precision, but one that gets the closest neighbor right scores higher on NDCG.

### 3. Distance Error
**Definition**: How much do the average distances differ from ground truth?

**Formula**: `|avg_predicted_distance - avg_true_distance| / avg_true_distance`

**Why it matters**: Shows if the algorithm preserves actual similarity relationships.

## Performance Metrics

### Memory Usage
- **Setup Memory**: Memory needed to build the index/data structure
- **Query Memory**: Additional memory used per search query
- **Total Memory**: Overall memory footprint

### CPU Time
- **Setup Time**: Time to build the index from vectors
- **Average Search Time**: Time per query (most important metric)
- **Total Search Time**: Cumulative time across all queries

## Apple-to-Apple Comparison Requirements

To ensure fair comparison between KNN methods:

### 1. Identical Dataset
```javascript
// All methods must use:
- Same 4096 vectors with 512 dimensions
- Same vector generation seed for reproducibility
- Same query vectors for testing
- Same distance metric (euclidean/cosine/manhattan)
```

### 2. Ground Truth Calculation
```javascript
// Exhaustive brute-force search provides the TRUE top-8:
function calculateGroundTruth(queryVector, allVectors, k=8) {
    const distances = allVectors.map(v => ({
        id: v.id,
        distance: euclideanDistance(queryVector, v.vector)
    }));
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, k); // TRUE top-k
}
```

### 3. Standardized Accuracy Measurement
```javascript
function measureAccuracy(predicted, groundTruth, k) {
    const predictedIds = new Set(predicted.slice(0, k).map(r => r.id));
    const trueIds = new Set(groundTruth.slice(0, k).map(r => r.id));
    
    const correctNeighbors = [...predictedIds].filter(id => trueIds.has(id));
    const precision = correctNeighbors.length / k;
    
    return precision; // This is the "accuracy percentage"
}
```

## Realistic Test Scenarios

### Dataset Characteristics
- **Size**: 4096 vectors (realistic production scale)
- **Dimensions**: 512 (common for embeddings)
- **Structure**: Clustered data (mimics real-world vector distributions)
- **Query Types**: Both cluster-center queries (easy) and random queries (hard)

### Expected Performance Ranges

| Algorithm Type | Typical Precision@8 | Search Time | Memory Usage |
|---------------|-------------------|-------------|--------------|
| Exhaustive Search | 100% | ~50ms | Low |
| High-Quality HNSW | 85-95% | 1-5ms | Medium |
| LSH | 70-85% | 1-3ms | High |
| Fast Approximation | 50-70% | 0.5-2ms | Low |
| Poor Implementation | 10-40% | Variable | Variable |

## How to Interpret Results

### Accuracy Trade-offs
- **95%+ Precision**: Excellent - nearly as good as exhaustive search
- **80-95% Precision**: Good - suitable for most applications
- **60-80% Precision**: Fair - acceptable for speed-critical applications
- **<60% Precision**: Poor - probably not suitable for production

### Speed Trade-offs
- **>20ms/query**: Too slow for real-time applications
- **5-20ms/query**: Acceptable for batch processing
- **1-5ms/query**: Good for real-time applications
- **<1ms/query**: Excellent for high-throughput systems

### Memory Trade-offs
- **<100MB**: Suitable for mobile/edge devices
- **100MB-1GB**: Acceptable for server applications
- **>1GB**: Only for high-memory specialized systems

## Real-World Application

When you say "I want to find the top 8 most similar vectors out of 4096," you're asking:

1. **Which 8 vectors are truly closest?** (Ground truth via exhaustive search)
2. **Which 8 does this algorithm return?** (Algorithm prediction)
3. **How many overlap?** (Accuracy measurement)

If 7 out of 8 returned vectors are in the true top-8, your algorithm has 87.5% accuracy - meaning it's finding the right neighbors 87.5% of the time.

## Why This Matters

- **Search engines**: Users expect relevant results, not just any results
- **Recommendation systems**: Suggesting the truly most similar items
- **Image recognition**: Finding the closest visual matches
- **Scientific computing**: Accurate similarity measurements for analysis

The "average accuracy" metric tells you: **"If I ask for the 8 most similar items, how often will this algorithm actually give me the real top 8?"** - which is exactly what you want to know!
