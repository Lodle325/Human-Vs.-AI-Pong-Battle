// ============================================================
// FITNESS.JS -- Tell the AI what good defending looks like
//
// THIS IS THE ONLY FILE YOU NEED TO MODIFY.
//
// computeFitness() is called for every brain in the population
// after each training round. It receives a stats object
// describing how the paddle performed against the test balls,
// and must return a number. Higher = better paddle.
//
// The best-scoring brains survive into the next generation.
//
// ============================================================
// WHAT YOU HAVE ACCESS TO
// ============================================================
//
//   paddle.hits            -- number of test balls intercepted
//   paddle.misses          -- number of test balls missed
//   paddle.totalFaced      -- total balls faced (hits + misses)
//   paddle.avgMissDistance -- average pixels off-centre when missing
//                             (0 = perfect, ~230 = nowhere near)
//
// Example: if hits=7 out of 10 total, and the 3 misses were
// on average 80px from the paddle centre, then:
//   hits = 7, misses = 3, totalFaced = 10, avgMissDistance = 80
//
// ============================================================
// EXPERIMENTS TO TRY
// ============================================================
//
//   -- Start with just: return paddle.hits;
//      Watch how quickly the AI improves. Is it fast enough?
//
//   -- Add a bonus for being close when missing:
//      return paddle.hits * 10 + (200 - paddle.avgMissDistance) * 0.05 * paddle.misses;
//      Does the AI learn faster or differently?
//
//   -- Try penalising misses heavily:
//      return paddle.hits * 10 - paddle.misses * 5;
//      Does the AI become more cautious?
//
//   -- What happens if you return paddle.misses * -1?
//      (punish misses only, no reward for hits)
//      Can the AI still learn?
//
// ============================================================

function computeFitness(paddle) {
  // ============================================================
  // SMART FITNESS — Rewards consistency, accuracy, and learning
  // without destroying genetic diversity.
  // ============================================================

  // 1. CORE REWARD: Every hit is valuable.
  let fitness = paddle.hits * 20;

  // 2. ACCURACY BONUS: Reward the *rate* of success.
  //    A paddle that hits 9/10 is much better than 5/10.
  if (paddle.totalFaced > 0) {
    let accuracy = paddle.hits / paddle.totalFaced;
    // Quadratic bonus: 100% accuracy = +1000, 50% = +250
    fitness += Math.pow(accuracy, 2) * 1000;
  }

  // 3. NEAR-MISS BONUS: Smooths the learning curve.
  //    If you miss, but miss by only 10px, you should get credit.
  if (paddle.misses > 0) {
    // avgMissDistance ranges from 0 to ~230.
    // We reward closeness: the closer, the better.
    let closenessFactor = Math.max(0, (200 - paddle.avgMissDistance) / 200);
    // Up to +5 points per miss for being extremely close.
    fitness += closenessFactor * 5 * paddle.misses;
  }

  // 4. MISS PENALTY: Reasonable, not crippling.
  //    Losing 10 points per miss means a perfect paddle gets a solid lead.
  fitness -= paddle.misses * 10;

  // 5. FACED BONUS: Encourages the AI to actively intercept.
  //    More faced balls = more data = more reliable fitness score.
  if (paddle.totalFaced > 10) {
    fitness += paddle.totalFaced * 0.5;
  } else {
    // Penalise if it faced too few balls (prevents gaming the system).
    fitness -= (10 - paddle.totalFaced) * 10;
  }

  // 6. PERFECTION BONUS: Big reward for flawless performance.
  if (paddle.misses === 0 && paddle.totalFaced >= 10) {
    fitness += 1500; // Enough to dominate selection.
  }

  // Ensure fitness is never negative (keeps selection stable).
  return Math.max(1, fitness);
}
