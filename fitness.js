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

  return 0;

}