=== THE EXACT DELPHI DPM MATH (verified byte-for-byte against the verified contract source — implement this faithfully, do NOT approximate) ===
Mechanism: L2 / Euclidean-norm dynamic parimutuel market. Solidity source DynamicParimutuelMath, verified on Gensyn mainnet explorer (Blockscout). Market impl 0x2fc709df2fb31d362355d1be3f81eeab8d238c5f, gateway 0x4e4e85c52E0F414cc67eE88d0C649Ec81698d700.

Let qᵢ = outstanding shares of outcome i (18-decimal fixed point). k = per-market liquidity constant (config.k). sumTerm36 = Σqᵢ².

COST / COLLATERAL (the invariant the pool tracks):
  C(q) = k · √(Σqᵢ²)          [pool == C, sumTerm36 == Σqᵢ²]

SPOT PRICE (marginal cost ∂C/∂qᵢ):
  pᵢ = k·qᵢ / √(Σqⱼ²) = k·qᵢ / √(sumTerm36)
  ⇒ Σpᵢ² = k²   (prices live on a sphere of radius k; they do NOT sum to 1)

IMPLIED PROBABILITY (price ≠ probability — this is critical, the public docs get this WRONG):
  πᵢ = qᵢ² / Σqⱼ² = (pᵢ/k)²     (Σπᵢ = 1)
  Implied probability is spot price SQUARED over k². (e.g. 0.51/share ≈ 26%, NOT 51%. The docs' "0.65 ≈ 65%" is wrong; it's 42%. Do not trust docs for math — this spec is from verified contract source.)

BUY Δ shares of outcome j (quoteBuyExactOut):
  newSumTerm36 = sumTerm36 + Δ·(2·qⱼ + Δ)        [= (qⱼ+Δ)² − qⱼ²]
  gross tokensIn = k·(√newSumTerm36 − √sumTerm36)
  then add tradingFee (2%): tokensIn = mulDivUp(gross, 1e18, 1e18 − fee)
  ROUNDING (always against the user): cost uses sqrtUp(new) − sqrtDown(current); tokensIn rounded UP.

SELL Δ shares of outcome j (quoteSellExactIn):
  newSumTerm36 = sumTerm36 − Δ·(2·qⱼ − Δ)        [reverses the buy]
  gross payout = k·(√sumTerm36 − √newSumTerm36)
  net = gross·(1e18 − fee)/1e18
  ROUNDING (always against the user): uses sqrtDown(new), sqrtUp(current); tokensOut rounded DOWN.

SETTLEMENT (winners split the pool pro-rata by shares — pure parimutuel):
  redeemerReward = mulDivDown(marketPool, redeemerWinningShares, unclaimedWinningShares)
  i.e. payout = pool · (yourWinningShares / totalWinningShares), rounded down. (unclaimedShares decrements as people redeem → order-independent.)

LIQUIDATION (expired/unsettled, no winner): you recover the spot value of held shares:
  liquidatorReward = k·Σ(held qᵢ)/√(Σqⱼ²), rounded down.

NOTE: the market-creator reward / 50% refund is NOT in DynamicParimutuelMath — it lives in the gateway contract (calculateInitialPoolAndRefund). Not needed for v1; ignore unless/until modeling creator economics.

All trading-time math (cost, price, probability, buy, sell) is verified to the digit. Settlement/liquidation confirmed from verified source. Implement in floating point for v1 but keep the formulas structurally identical; the 18-dec fixed-point + against-the-user rounding can be matched later if byte-parity with the chain is ever needed.