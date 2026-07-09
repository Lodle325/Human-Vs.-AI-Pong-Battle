// ============================================================
// GAME.JS -- Pong physics, neural network, genetic trainer
// READ ONLY -- your work goes in fitness.js
// ============================================================

// ============================================================
// CONSTANTS
// ============================================================
var CANVAS_W = 660;
var CANVAS_H = 460;
var PADDLE_W = 12;
var PADDLE_H = 80;
var BALL_R = 7;
var PLAYER_X = 22;
var AI_X_POS = CANVAS_W - PADDLE_W - 22;
var PADDLE_SPD = 5.2;
var BALL_SPEED = 4.5;
var BALL_MAX_SPD = 9.0;
var WIN_SCORE = 7;

// How many generations the AI trains after each scored point.
// Higher = AI improves faster but might improve too quickly for students.
var GENS_PER_POINT = 100;

// Neural network architecture:
// 5 inputs -> 8 hidden -> 1 output
var NN_ARCH = [5, 8, 1];

// Genetic algorithm settings
var POP_SIZE = 20;
var ELITE_N = 4;
var MUT_RATE = 0.12;
var MUT_STRENGTH = 0.35;

// ============================================================
// TEST BALL TRAJECTORIES
// Pre-computed ball paths used during AI training.
// The AI trains by seeing how well it intercepts each of these.
// ============================================================
var TEST_BALLS = (function () {
  var balls = [];
  var angles = [-55, -40, -25, -10, 0, 10, 25, 40, 55];
  var startYs = [
    CANVAS_H * 0.15,
    CANVAS_H * 0.35,
    CANVAS_H * 0.5,
    CANVAS_H * 0.65,
    CANVAS_H * 0.85,
  ];
  angles.forEach(function (deg) {
    startYs.forEach(function (sy) {
      var rad = (deg * Math.PI) / 180;
      balls.push({
        x: CANVAS_W / 2,
        y: sy,
        vx: BALL_SPEED * Math.cos(rad),
        vy: BALL_SPEED * Math.sin(rad),
      });
    });
  });
  return balls;
})();

// ============================================================
// NEURAL NETWORK
//
// Takes 5 inputs and produces 1 output (-1 to 1).
//   > +0.2  = move paddle up
//   < -0.2  = move paddle down
//   otherwise = stay
//
// Inputs:
//   0: ball x (0 to 1)
//   1: ball y (0 to 1)
//   2: ball x velocity (normalised)
//   3: ball y velocity (normalised)
//   4: AI paddle centre y (0 to 1)
// ============================================================
function NeuralNetwork(sizes) {
  sizes = sizes || NN_ARCH;
  this.sizes = sizes;
  this.weights = [];
  this.biases = [];
  this.fitness = 0;

  for (var i = 1; i < sizes.length; i++) {
    var lw = [],
      lb = [];
    for (var j = 0; j < sizes[i]; j++) {
      var row = [];
      for (var k = 0; k < sizes[i - 1]; k++) {
        row.push((Math.random() * 2 - 1) * 0.6);
      }
      lw.push(row);
      lb.push((Math.random() * 2 - 1) * 0.1);
    }
    this.weights.push(lw);
    this.biases.push(lb);
  }
}

NeuralNetwork.prototype.forward = function (inputs) {
  var cur = inputs.slice();
  for (var l = 0; l < this.weights.length; l++) {
    var next = [];
    for (var j = 0; j < this.weights[l].length; j++) {
      var s = this.biases[l][j];
      for (var k = 0; k < cur.length; k++) s += cur[k] * this.weights[l][j][k];
      next.push(Math.tanh(s));
    }
    cur = next;
  }
  return cur;
};

NeuralNetwork.prototype.clone = function () {
  var nn = new NeuralNetwork(this.sizes);
  nn.weights = this.weights.map(function (l) {
    return l.map(function (r) {
      return r.slice();
    });
  });
  nn.biases = this.biases.map(function (l) {
    return l.slice();
  });
  return nn;
};

NeuralNetwork.prototype.mutate = function (rate, strength) {
  rate = rate !== undefined ? rate : MUT_RATE;
  strength = strength !== undefined ? strength : MUT_STRENGTH;
  this.weights = this.weights.map(function (l) {
    return l.map(function (r) {
      return r.map(function (w) {
        return Math.random() < rate
          ? w + (Math.random() * 2 - 1) * strength
          : w;
      });
    });
  });
  this.biases = this.biases.map(function (l) {
    return l.map(function (b) {
      return Math.random() < rate ? b + (Math.random() * 2 - 1) * strength : b;
    });
  });
  return this;
};

NeuralNetwork.prototype.crossover = function (other) {
  var child = this.clone();
  child.weights = child.weights.map(function (l, i) {
    return l.map(function (r, j) {
      return r.map(function (w, k) {
        return Math.random() < 0.5 ? w : other.weights[i][j][k];
      });
    });
  });
  child.biases = child.biases.map(function (l, i) {
    return l.map(function (b, j) {
      return Math.random() < 0.5 ? b : other.biases[i][j];
    });
  });
  return child;
};

// ============================================================
// TRAINING SIMULATION
//
// Runs entirely in JavaScript with no rendering.
// Fires every test ball at a simulated paddle and measures
// how well each neural network intercepts them.
// Called after each scored point.
// ============================================================
var generation = 0;
var brains = [];
var activeBrain = null; // the best brain, used in live play
var bestFitness = 0;
var fitnessHistory = [];

function initBrains() {
  generation = 0;
  brains = [];
  bestFitness = 0;
  fitnessHistory = [];
  activeBrain = null;
  for (var i = 0; i < POP_SIZE; i++) {
    brains.push(new NeuralNetwork(NN_ARCH));
  }
}

// Simulate a single ball against a single neural network.
// The ball is fired from the given starting position.
// Returns { hit: bool, missDistance: number }
function simOneRally(ball, brain) {
  var bx = ball.x,
    by = ball.y;
  var bvx = ball.vx,
    bvy = ball.vy;
  var py = CANVAS_H / 2 - PADDLE_H / 2; // paddle starts centred

  for (var step = 0; step < 500; step++) {
    // Wall bounces
    if (by - BALL_R <= 0) bvy = Math.abs(bvy);
    if (by + BALL_R >= CANVAS_H) bvy = -Math.abs(bvy);

    bx += bvx;
    by += bvy;

    // Neural network decides how to move the paddle
    var out = brain.forward([
      bx / CANVAS_W,
      by / CANVAS_H,
      bvx / BALL_MAX_SPD,
      bvy / BALL_MAX_SPD,
      (py + PADDLE_H / 2) / CANVAS_H,
    ])[0];

    if (out > 0.2) py -= PADDLE_SPD;
    else if (out < -0.2) py += PADDLE_SPD;
    py = Math.max(0, Math.min(CANVAS_H - PADDLE_H, py));

    // Has the ball reached the AI paddle?
    if (bx + BALL_R >= AI_X_POS) {
      var centre = py + PADDLE_H / 2;
      var miss = Math.abs(by - centre);
      return { hit: miss <= PADDLE_H / 2 + BALL_R, missDistance: miss };
    }

    if (bx < 0) break;
  }
  return { hit: false, missDistance: CANVAS_H };
}

// Score all brains in the population against TEST_BALLS,
// evolve the best ones, and update activeBrain.
function runOneGeneration() {
  generation++;

  brains.forEach(function (brain) {
    var hits = 0,
      misses = 0,
      totalMissDist = 0;

    TEST_BALLS.forEach(function (ball) {
      var r = simOneRally(ball, brain);
      if (r.hit) {
        hits++;
      } else {
        misses++;
        totalMissDist += r.missDistance;
      }
    });

    // Build the stats object passed to computeFitness()
    var stats = {
      hits: hits,
      misses: misses,
      totalFaced: TEST_BALLS.length,
      avgMissDistance: misses > 0 ? totalMissDist / misses : 0,
    };

    brain.fitness = computeFitness(stats);
  });

  // Sort best to worst
  brains.sort(function (a, b) {
    return b.fitness - a.fitness;
  });

  var top = brains[0].fitness;
  fitnessHistory.push(top);

  if (top > bestFitness) {
    bestFitness = top;
    activeBrain = brains[0].clone();
  }

  // Elites survive; fill the rest with mutated offspring
  var next = [];
  for (var i = 0; i < ELITE_N; i++) next.push(brains[i].clone());
  while (next.length < POP_SIZE) {
    var ia = Math.floor(Math.random() * ELITE_N);
    var ib = Math.floor(Math.random() * ELITE_N);
    var child = brains[ia].crossover(brains[ib]).mutate();
    next.push(child);
  }
  brains = next;
}

// Called after each scored point -- runs GENS_PER_POINT generations
function trainAI() {
  for (var g = 0; g < GENS_PER_POINT; g++) {
    runOneGeneration();
  }
}

// Ask the active brain what to do this frame
function getAIMove(ballObj, aiPadObj) {
  if (!activeBrain) return 0;
  return activeBrain.forward([
    ballObj.x / CANVAS_W,
    ballObj.y / CANVAS_H,
    ballObj.vx / BALL_MAX_SPD,
    ballObj.vy / BALL_MAX_SPD,
    (aiPadObj.y + PADDLE_H / 2) / CANVAS_H,
  ])[0];
}

// ============================================================
// GAME STATE
// ============================================================
var state = "waiting"; // 'waiting' | 'playing' | 'scored' | 'gameover'
var playerScore = 0;
var aiScore = 0;
var lastScorer = null; // 'player' | 'ai'
var scoredTimer = 0;

var ballObj = { x: 0, y: 0, vx: 0, vy: 0, speed: BALL_SPEED };

var playerPad = { x: PLAYER_X, y: 0 };
var aiPad = { x: AI_X_POS, y: 0 };

var keys = {};
document.addEventListener("keydown", function (e) {
  keys[e.key] = true;
  if (e.key === " ") {
    if (state === "waiting") serveBall();
    if (state === "gameover") restartGame();
    e.preventDefault();
  }
  if (["ArrowUp", "ArrowDown", "w", "s", "W", "S"].indexOf(e.key) !== -1) {
    e.preventDefault();
  }
});
document.addEventListener("keyup", function (e) {
  keys[e.key] = false;
});

// ============================================================
// GAME LOGIC
// ============================================================
function resetPositions() {
  playerPad.y = CANVAS_H / 2 - PADDLE_H / 2;
  aiPad.y = CANVAS_H / 2 - PADDLE_H / 2;
  ballObj.x = CANVAS_W / 2;
  ballObj.y = CANVAS_H / 2;
  ballObj.vx = 0;
  ballObj.vy = 0;
  ballObj.speed = BALL_SPEED;
}

function serveBall() {
  if (state !== "waiting") return;
  state = "playing";

  // Launch toward whoever just lost (or random at start)
  var dir = lastScorer === "ai" ? -1 : 1;
  var angle = ((Math.random() * 50 - 25) * Math.PI) / 180;
  ballObj.speed = BALL_SPEED;
  ballObj.vx = dir * ballObj.speed * Math.cos(angle);
  ballObj.vy = ballObj.speed * Math.sin(angle);
}

function updateBall() {
  ballObj.x += ballObj.vx;
  ballObj.y += ballObj.vy;

  // Top / bottom wall
  if (ballObj.y - BALL_R <= 0) {
    ballObj.y = BALL_R;
    ballObj.vy = Math.abs(ballObj.vy);
  }
  if (ballObj.y + BALL_R >= CANVAS_H) {
    ballObj.y = CANVAS_H - BALL_R;
    ballObj.vy = -Math.abs(ballObj.vy);
  }

  // Player paddle hit
  if (
    ballObj.vx < 0 &&
    ballObj.x - BALL_R <= PLAYER_X + PADDLE_W &&
    ballObj.x - BALL_R >= PLAYER_X - 2 &&
    ballObj.y + BALL_R >= playerPad.y &&
    ballObj.y - BALL_R <= playerPad.y + PADDLE_H
  ) {
    applyPaddleHit(playerPad, true);
  }

  // AI paddle hit
  if (
    ballObj.vx > 0 &&
    ballObj.x + BALL_R >= AI_X_POS &&
    ballObj.x + BALL_R <= AI_X_POS + PADDLE_W + 2 &&
    ballObj.y + BALL_R >= aiPad.y &&
    ballObj.y - BALL_R <= aiPad.y + PADDLE_H
  ) {
    applyPaddleHit(aiPad, false);
  }

  // Scoring
  if (ballObj.x - BALL_R < 0) {
    aiScore++;
    lastScorer = "ai";
    handleScore();
  }
  if (ballObj.x + BALL_R > CANVAS_W) {
    playerScore++;
    lastScorer = "player";
    handleScore();
  }
}

function applyPaddleHit(pad, isPlayer) {
  // Direction reversal
  ballObj.vx = isPlayer ? Math.abs(ballObj.vx) : -Math.abs(ballObj.vx);

  // Angle based on contact point on paddle
  var rel = (ballObj.y - (pad.y + PADDLE_H / 2)) / (PADDLE_H / 2);
  var angle = rel * 0.75;
  ballObj.speed = Math.min(ballObj.speed + 0.3, BALL_MAX_SPD);
  ballObj.vy = Math.sin(angle) * ballObj.speed;

  // Push ball clear of paddle to prevent sticking
  if (isPlayer) ballObj.x = PLAYER_X + PADDLE_W + BALL_R + 1;
  else ballObj.x = AI_X_POS - BALL_R - 1;
}

function handleScore() {
  if (playerScore >= WIN_SCORE || aiScore >= WIN_SCORE) {
    state = "gameover";
  } else {
    state = "scored";
    scoredTimer = 100; // ~1.7 seconds at 60fps
    trainAI();
    resetPositions();
  }
}

function restartGame() {
  playerScore = 0;
  aiScore = 0;
  lastScorer = null;
  initBrains();
  resetPositions();
  state = "waiting";
}

function updateGame() {
  if (state === "scored") {
    scoredTimer--;
    if (scoredTimer <= 0) state = "waiting";
    return;
  }

  if (state !== "playing") return;

  // Player paddle movement
  if (keys["ArrowUp"] || keys["w"] || keys["W"]) playerPad.y -= PADDLE_SPD;
  if (keys["ArrowDown"] || keys["s"] || keys["S"]) playerPad.y += PADDLE_SPD;
  playerPad.y = Math.max(0, Math.min(CANVAS_H - PADDLE_H, playerPad.y));

  // AI paddle movement
  var aiOut = getAIMove(ballObj, aiPad);
  if (aiOut > 0.2) aiPad.y -= PADDLE_SPD;
  else if (aiOut < -0.2) aiPad.y += PADDLE_SPD;
  aiPad.y = Math.max(0, Math.min(CANVAS_H - PADDLE_H, aiPad.y));

  updateBall();
}

// ============================================================
// RENDERING
// ============================================================

// AI paddle colour transitions from grey -> orange as it trains
function getAIColor(alpha) {
  var t = Math.min(1, generation / 50);
  var r = Math.round(110 + 145 * t);
  var g = Math.round(110 - 60 * t);
  var b = Math.round(110 - 88 * t);
  if (alpha !== undefined)
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  return "rgb(" + r + "," + g + "," + b + ")";
}

function drawGame(ctx) {
  var W = CANVAS_W,
    H = CANVAS_H;

  // Background
  ctx.fillStyle = "#0e0e0e";
  ctx.fillRect(0, 0, W, H);

  // Centre dashed line
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  ctx.beginPath();
  ctx.moveTo(W / 2, 0);
  ctx.lineTo(W / 2, H);
  ctx.stroke();
  ctx.setLineDash([]);

  // Scores
  ctx.font = "bold 56px monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(34,197,94,0.55)";
  ctx.fillText(playerScore, W / 4, 12);
  ctx.fillStyle = getAIColor(0.55);
  ctx.fillText(aiScore, (3 * W) / 4, 12);

  // Labels
  ctx.font = "10px monospace";
  ctx.fillStyle = "rgba(255,255,255,0.2)";
  ctx.fillText("YOU", W / 4, H - 18);
  ctx.fillText("AI  GEN " + generation, (3 * W) / 4, H - 18);

  // Player paddle (green)
  ctx.fillStyle = "#22c55e";
  ctx.shadowColor = "#22c55e";
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.roundRect(playerPad.x, playerPad.y, PADDLE_W, PADDLE_H, 3);
  ctx.fill();

  // AI paddle (colour tracks generation)
  ctx.fillStyle = getAIColor();
  ctx.shadowColor = getAIColor();
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.roundRect(aiPad.x, aiPad.y, PADDLE_W, PADDLE_H, 3);
  ctx.fill();

  ctx.shadowBlur = 0;

  // Ball
  ctx.fillStyle = "#ffffff";
  ctx.shadowColor = "#ffffff";
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.arc(ballObj.x, ballObj.y, BALL_R, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // ---- Overlays ----
  if (state === "waiting") {
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = "15px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PRESS SPACE TO SERVE", W / 2, H - 34);

    if (generation === 0) {
      ctx.fillStyle = "rgba(255,255,255,0.22)";
      ctx.font = "12px monospace";
      ctx.fillText(
        "AI is untrained — beat it before it learns!",
        W / 2,
        H / 2 + 44,
      );
    }
  }

  if (state === "scored") {
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = lastScorer === "player" ? "#22c55e" : getAIColor();
    ctx.font = "bold 20px monospace";
    ctx.fillText(
      lastScorer === "player" ? "YOU SCORED" : "AI SCORED",
      W / 2,
      H / 2 - 14,
    );
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "12px monospace";
    ctx.fillText("Training...  generation " + generation, W / 2, H / 2 + 14);
  }

  if (state === "gameover") {
    ctx.fillStyle = "rgba(0,0,0,0.72)";
    ctx.fillRect(0, 0, W, H);

    var playerWon = playerScore >= WIN_SCORE;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = playerWon ? "#22c55e" : getAIColor();
    ctx.font = "bold 34px monospace";
    ctx.fillText(playerWon ? "YOU WIN!" : "AI WINS!", W / 2, H / 2 - 36);

    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "16px monospace";
    ctx.fillText(playerScore + "  —  " + aiScore, W / 2, H / 2 + 4);

    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "12px monospace";
    ctx.fillText("AI reached generation " + generation, W / 2, H / 2 + 32);

    ctx.fillStyle = "rgba(255,255,255,0.2)";
    ctx.fillText("SPACE to play again", W / 2, H / 2 + 62);
  }
}
