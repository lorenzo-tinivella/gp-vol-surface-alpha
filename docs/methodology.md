# GP Vol Surface Arbitrage — Project Reference

**Alpha Research | Options | Gaussian Process + Bayesian Optimization**

---

## Overview

Il progetto cattura inefficienze sistematiche nella volatility surface delle opzioni equity
usando un Gaussian Process come modello non-parametrico di riferimento e Bayesian Optimization
per calibrazione e ottimizzazione della soglia di trading. L'alpha economico deriva dalla
discrepanza tra la vol implicita di mercato e la stima GP — aggiustata per liquidità,
consistenza e costi di transazione — in presenza di un documentato Variance Risk Premium.

---

## Punto 1 — Raccolta Dati e Calcolo dell'Implied Volatility

### Cosa facciamo
Scarichiamo le options chain giornaliere (SPX o singolo titolo liquido) e invertiamo la
formula di Black-Scholes per ottenere la Implied Volatility (IV) per ogni coppia
(strike K, scadenza T). Puliamo i dati rimuovendo contratti con open interest basso e
prezzi fuori dai bound bid-ask.

### Idee matematiche
- **Formula di Black-Scholes:** C = S·N(d₁) − K·e^(−rT)·N(d₂)
  dove d₁ = [ln(S/K) + (r + σ²/2)T] / σ√T
- **Inversione numerica:** IV = argmin_σ |C_BS(σ) − C_market|
  risolta con il metodo di Brent (root-finding bracketed, convergenza garantita)
- **Moneyness log-forward:** k = ln(K / F) dove F = S·e^(rT)
  standardizza lo strike rispetto alla struttura forward

### Idee economiche
- L'IV non è una volatilità storica — è la volatilità che, se inserita in BS, replica
  il prezzo di mercato. Incorpora le aspettative del mercato *e* un premio al rischio.
- La costruzione della superficie richiede dati puliti: contratti illiquidi hanno
  bid-ask enormi che rendono l'IV computata inaffidabile e non tradable.
- Il passaggio a moneyness log-forward rimuove la dipendenza dal livello spot,
  rendendo la superficie confrontabile nel tempo.

### Librerie
`yfinance`, `py_vollib`, `scipy.optimize.brentq`

### Riferimenti bibliografici
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
  Journal of Political Economy, 81(3), 637–654.
- Breeden, D. T., & Litzenberger, R. H. (1978). *Prices of State-Contingent Claims
  Implicit in Option Prices.* Journal of Business, 51(4), 621–651.

---

## Punto 2 — Costruzione della Superficie e Vincoli No-Arbitrage

### Cosa facciamo
Organizziamo i punti (k, T) → IV in una griglia sparsa e applichiamo controlli
di arbitraggio statico. I punti che violano i vincoli vengono rimossi o flaggati
prima di fittare il GP.

### Idee matematiche
- **Butterfly arbitrage:** ∂²C/∂K² ≥ 0 per ogni K fissato — la curva dei prezzi
  rispetto allo strike deve essere convessa. Equivale a densità di probabilità
  risk-neutral non-negativa.
- **Calendar spread arbitrage:** ∂C/∂T ≥ 0 per ogni K fissato — un'opzione con
  scadenza più lontana non può valere meno di una con scadenza vicina (stesso strike).
- **Condizione di Breeden-Litzenberger:** q(S_T) = e^(rT) · ∂²C/∂K²
  permette di estrarre la distribuzione risk-neutral dei prezzi futuri dalla superficie.

### Idee economiche
- Le violazioni di questi vincoli rappresentano opportunità di arbitraggio statico
  senza rischio. In mercati efficienti non dovrebbero esistere — se le troviamo,
  sono quasi sempre artefatti di dati illiquidi o errori di quotazione.
- Rimuovere questi punti non impoverisce il dataset: sono prezzi non informativi
  che distorcerebbero il fit del GP.

### Riferimenti bibliografici
- Dupire, B. (1994). *Pricing with a Smile.* Risk, 7(1), 18–20.
- Fengler, M. R. (2009). *Arbitrage-Free Smoothing of the Implied Volatility Surface.*
  Quantitative Finance, 9(4), 417–428.

---

## Punto 3 — Gaussian Process sulla Superficie di Volatilità

### Cosa facciamo
Fittiamo un Gaussian Process Regressor sulla griglia sparsa di punti puliti.
Input: X = [k, log(T)]. Output: distribuzione su IV, ovvero una media predetta
μ_GP(k,T) e una deviazione standard σ_GP(k,T) per ogni punto della superficie.

### Idee matematiche
- **GP come prior su funzioni:** f ~ GP(m(x), κ(x,x'))
  dove m è la funzione media (tipicamente zero) e κ è il kernel di covarianza.
- **Kernel composito:** κ = RBF(l₁) + Matérn_5/2(l₂) + WhiteNoise(σ_n)
  - RBF cattura la smoothness globale della superficie
  - Matérn cattura irregolarità locali (meno smooth dell'RBF)
  - WhiteNoise modella il rumore di bid-ask
- **Posterior gaussiano:** dopo aver osservato i dati, il posterior è analitico:
  μ_GP(x*) = κ(x*, X) · [κ(X,X) + σ_n²I]⁻¹ · y
  σ²_GP(x*) = κ(x*,x*) − κ(x*,X) · [κ(X,X) + σ_n²I]⁻¹ · κ(X,x*)
- **Ottimizzazione degli iperparametri** (lunghezze di scala l₁, l₂ e noise σ_n):
  massimizzazione della log-marginal likelihood:
  log p(y|X,θ) = −½ yᵀ K⁻¹ y − ½ log|K| − n/2 log(2π)

### Idee economiche
- Il GP è un interpolatore ottimale in media quadratica: usa tutti i prezzi
  osservati come vincoli reciproci per stimare la IV in ogni punto della superficie.
- σ_GP(k,T) è una misura diretta di illiquidità locale: alta dove ci sono pochi
  contratti scambiati, bassa dove il mercato è denso e informativo.
- Una deviazione significativa tra IV_market e μ_GP segnala che quell'opzione è
  prezzata in modo inconsistente con i suoi vicini — potenziale mispricing.

### Librerie
`sklearn.gaussian_process`, `GPy`, `botorch` (pytorch-based)

### Riferimenti bibliografici
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.*
  MIT Press.
- Cont, R., & da Fonseca, J. (2002). *Dynamics of Implied Volatility Surfaces.*
  Quantitative Finance, 2(1), 45–60.
- Cousin, A., Maatouk, H., & Rullière, D. (2016). *Kriging of Financial Term-Structures.*
  European Journal of Operational Research, 255(2), 631–648.

---

## Punto 4 — Modello SVI come Baseline Parametrica (calibrato con BO)

### Cosa facciamo
Calibriamo il modello SVI (Stochastic Volatility Inspired) sulla superficie giornaliera
usando Bayesian Optimization invece di grid search. SVI produce una superficie parametrica
arbitrage-free che serve come termine di confronto per il GP.

### Idee matematiche
- **Forma funzionale SVI:**
  σ²_SVI(k) = a + b · [ρ(k − m) + √((k − m)² + ξ²)]
  dove θ = (a, b, ρ, m, ξ) sono i 5 parametri da calibrare (per ogni scadenza T).
  - a: livello di varianza totale
  - b: pendenza dell'ala (ATM vol)
  - ρ ∈ (−1,1): correlazione skew (asimmetria)
  - m: centro della curva (ATM offset)
  - ξ: smoothness dell'ala (curvatura)
- **Bayesian Optimization:**
  Costruisce un GP surrogate model sulla funzione di loss L(θ) = Σ(IV_market − IV_SVI)²
  e usa una acquisition function per scegliere dove valutare L(θ) successivamente:
  - Expected Improvement: EI(θ) = E[max(L(θ*) − L(θ), 0)]
  - Upper Confidence Bound: UCB(θ) = μ(θ) + β·σ(θ)
  BO trova il minimo in ~50 valutazioni vs ~10⁵ della grid search.
- **Condizioni no-arbitrage per SVI:**
  b ≥ 0, |ρ| < 1, ξ > 0, a + b·ξ·√(1−ρ²) ≥ 0

### Idee economiche
- SVI rappresenta la "prior parametrica" del market maker: è il modello che tipicamente
  usano per interpolare la superficie sulle zone illiquide.
- Usando BO invece di grid search non è solo efficienza computazionale: riduce il
  rischio di overfittare i parametri agli stessi dati che useremo per il segnale.
- Il confronto GP vs SVI è al cuore del segnale: dove il modello flessibile (GP)
  e quello rigido (SVI) concordano su un mispricing, il segnale è molto più credibile.

### Librerie
`scikit-optimize (skopt)`, `optuna`, `botorch`

### Riferimenti bibliografici
- Gatheral, J., & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces.*
  Quantitative Finance, 14(1), 59–71.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization
  of Machine Learning Algorithms.* NeurIPS, 25, 2951–2959.
- Frazier, P. I. (2018). *A Tutorial on Bayesian Optimization.* arXiv:1807.02811.

---

## Punto 5 — Calendar Filter e Gestione degli Eventi

### Cosa facciamo
Prima di calcolare qualsiasi segnale, flagghiamo le opzioni la cui scadenza cade
entro 3 giorni da un evento noto (earnings, FOMC, CPI release). Il moltiplicatore
di calendario cal(T) = 0 per queste opzioni, indipendentemente dal segnale GP.

### Idee matematiche
- **Definizione formale:** cal(T) = 𝟙[min_{e ∈ Events} |T − e| > δ]
  dove δ = 3 giorni e Events include FOMC, earnings date, macro releases.
- **Event vol decomposition:**
  σ²_total = σ²_daily · (T − t) + σ²_event · 𝟙[event ∈ [t, T]]
  La presenza di un evento spiega sistematicamente parte della IV totale.

### Idee economiche
- Un'opzione che ingloba un earnings non è misprezzata se la sua IV è alta:
  sta correttamente prezzando l'event risk. Non è inconsistente con i vicini
  — è in una categoria diversa.
- Questo filtro è l'unico che rimane binario: non c'è gradazione possibile.
  Un'opzione che copre un evento è fundamentalmente diversa dalle altre.
- La logica è quella di separare vol strutturale (quello che vogliamo tradare)
  da vol di evento (correttamente prezzata, non tradable).

### Riferimenti bibliografici
- Garleanu, N., Pedersen, L. H., & Poteshman, A. M. (2009). *Demand-Based Option Pricing.*
  Review of Financial Studies, 22(10), 4259–4299.

---

## Punto 6 — Composite Scoring Function

### Cosa facciamo
Convertiamo i quattro filtri in moltiplicatori continui e li combiniamo in uno score
unico per ogni opzione. Il trade avviene solo se lo score supera una soglia τ ottimizzata
con BO (vedi Punto 7).

### Idee matematiche
- **Z-score della deviazione:**
  z(k,T) = [IV_market(k,T) − μ_GP(k,T)] / σ_GP(k,T)
  Misura la deviazione in unità di incertezza del modello.

- **Confidence (dall'uncertainty GP):**
  conf(k,T) = 1 / (1 + σ_GP(k,T))    ∈ (0, 1]

- **Consistency (accordo GP e SVI):**
  cons(k,T) = 𝟙[sign(IV_market − μ_GP) = sign(IV_market − μ_SVI)]

- **Net deviation (aggiustata per bid-ask):**
  Δ_net(k,T) = max(|IV_market − μ_GP| − spread/2, 0)

- **Composite score:**
  score(k,T) = z · cal · conf · cons · Δ_net

  Proprietà:
  - score = 0 se qualunque moltiplicatore è zero
  - score è continuo → ranking preciso delle opportunità
  - size della posizione ∝ score (position sizing naturale)

### Idee economiche
- Ogni moltiplicatore ha una precisa interpretazione economica:
  - z: quanto è grande il mispricing in termini statistici
  - cal: il mispricing non è event vol (non tradable)
  - conf: il mercato in quella zona è abbastanza liquido da rendere il GP affidabile
  - cons: due modelli indipendenti concordano → il mercato ha torto, non i modelli
  - Δ_net: il mispricing supera i costi di transazione → il trade è profittevole
- Il composite score non è uno strumento statistico — è una formalizzazione
  dell'analisi economica che farebbe un trader esperto guardando la superficie.

---

## Punto 7 — Walk-Forward Backtest con BO per la Soglia

### Cosa facciamo
Usiamo BO per trovare la soglia ottimale τ su finestre rolling di training,
validando sempre su dati mai visti. Ogni finestra produce un τ*, usato nel
periodo successivo out-of-sample.

### Idee matematiche
- **Schema walk-forward:**
  Per ogni t = t₀, t₀+Δ, t₀+2Δ, ...
  - Train: dati da [t − W, t]  →  BO trova τ*(t)
  - Test:  dati da [t, t + Δ]  →  applica τ*(t), misura Sharpe OOS
  Tipicamente W = 252 giorni, Δ = 63 giorni (trimestrale).

- **BO per ottimizzazione di τ:**
  max_τ  Sharpe_OOS(τ)  su  τ ∈ [τ_min, τ_max]
  Usando GP surrogate + UCB acquisition function.
  ~30–50 valutazioni sufficienti (vs 500+ grid search).

- **Deflated Sharpe Ratio (DSR):**
  Corregge il Sharpe per il numero di configurazioni testate:
  DSR = SR · [1 − γ(skewness, kurtosis)] / √(V_trials)
  Previene il reporting di Sharpe inflati da selezione multipla.

- **Metriche di performance:**
  - Sharpe annualizzato: SR = E[R_daily] / σ[R_daily] · √252
  - Max drawdown: MDD = max_{t≤s} [V(t) − V(s)] / V(t)
  - Calmar ratio: SR / MDD
  - Hit rate: % trade con P&L > 0
  - Signal half-life: stima da autocorrelazione AR(1) del segnale

### Idee economiche
- Il walk-forward replica la sola situazione realistica: un gestore che nel passato
  avrebbe potuto usare solo i dati disponibili fino a quel momento.
- Senza walk-forward, qualsiasi backtest è inconsistente con la realtà perché
  usa informazioni future (look-ahead bias).
- Il DSR protegge dal data snooping: se si testano 50 configurazioni, la probabilità
  di trovare una buona per puro caso è alta. Il DSR corregge questa inflazione.

### Riferimenti bibliografici
- Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.*
  Journal of Portfolio Management, 40(5), 94–107.
- White, H. (2000). *A Reality Check for Data Snooping.*
  Econometrica, 68(5), 1097–1126.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

---

## Punto 8 — Delta-Hedging e Isolamento del Vol Alpha

### Cosa facciamo
Ogni posizione viene delta-hedgiata quotidianamente: compriamo o vendiamo la
quantità di azioni necessaria a rendere il portafoglio delta-neutro. Questo
elimina l'esposizione direzionale e isola il P&L alla sola componente di volatilità.

### Idee matematiche
- **Delta BS:**
  Δ_call = N(d₁),   Δ_put = N(d₁) − 1
  dove d₁ = [ln(S/K) + (r + σ²/2)T] / σ√T

- **P&L decomposition (Carr-Madan):**
  dΠ = (Γ/2)(dS)² · (σ²_realized − σ²_implied) · dt + residuo
  Il P&L di una posizione delta-hedgiata è proporzionale alla differenza tra
  volatilità realizzata e volatilità implicita pagata all'acquisto.

- **Gamma P&L giornaliero:**
  PnL_t ≈ ½ · Γ · S² · (r²_t − σ²_IV) · Δt
  dove r_t è il return giornaliero del sottostante.

- **Vega exposure:**
  Vega = ∂V/∂σ = S · N'(d₁) · √T
  Residua dopo il delta-hedge: è l'esposizione ai movimenti di vol implicita.

### Idee economiche
- Senza delta-hedge, il P&L della strategia include una grande componente
  direzionale (beta all'equity) che non è l'alpha che cerchiamo.
- Il delta-hedge trasforma la nostra posizione in un "bet puro" sulla vol:
  guadagnamo se σ_realized > σ_implied (abbiamo comprato vol economica)
  o se σ_implied scende verso il valore GP (il mispricing si corregge).
- Economicamente, il P&L post-hedge misura esclusivamente se avevamo ragione
  sulla val del GP — è il test diretto della nostra tesi di alpha.

### Riferimenti bibliografici
- Bakshi, G., & Kapadia, N. (2003). *Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.* Review of Financial Studies, 16(2), 527–566.
- Carr, P., & Wu, L. (2009). *Variance Risk Premiums.*
  Review of Financial Studies, 22(3), 1311–1341.

---

## Punto 9 — Analisi del Segnale e Diagnostica

### Cosa facciamo
Verifichiamo che il segnale abbia caratteristiche compatibili con alpha reale
e non con overfitting o noise: decadimento temporale consistente con arbitraggio,
consistenza cross-sezionale, stabilità nel tempo.

### Idee matematiche
- **Signal decay (half-life):** stimiamo la velocità di correzione del mispricing
  con un modello AR(1) sulla deviazione:
  Δ_t = α · Δ_{t−1} + ε    →    half-life = −log(2) / log(α)
  Un half-life di 2–5 giorni è compatibile con arbitraggio di mercato maker.

- **Stabilità cross-sezionale:**
  Correlazione del segnale su sottostanti diversi nella stessa giornata.
  Un segnale cross-sectionally consistente è molto meno probabile sia noise.

- **Information coefficient (IC):**
  IC_t = corr(score_t, return_vol_t+1)
  Misura il potere predittivo del segnale. IC > 0.05 è considerato buono
  in vol trading.

### Idee economiche
- Un half-life troppo corto (< 1 giorno) suggerirebbe che il segnale è
  microstructure noise non tradable con dati EOD.
- Un half-life troppo lungo (> 20 giorni) suggerirebbe che non è un mispricing
  ma una struttura economica persistente che non si corregge facilmente.
- Il range 2–10 giorni è compatibile con la velocità di ricalibrazione dei
  market maker e con la persistenza della pressione della domanda istituzionale.

### Riferimenti bibliografici
- Cont, R., & da Fonseca, J. (2002). *Dynamics of Implied Volatility Surfaces.*
  Quantitative Finance, 2(1), 45–60.

---

## Stack Tecnologico Completo

| Componente          | Libreria principale          | Alternativa           |
|---------------------|------------------------------|-----------------------|
| Dati options        | `yfinance`                   | `polygon.io` API      |
| Implied Vol         | `py_vollib`                  | `scipy` + BS custom   |
| Gaussian Process    | `sklearn.gaussian_process`   | `GPy`, `botorch`      |
| Bayesian Opt.       | `scikit-optimize`            | `optuna`              |
| Backtest            | `vectorbt`                   | custom `pandas`       |
| Visualizzazione     | `plotly` (3D surface)        | `matplotlib`          |
| Delta hedge         | custom + `py_vollib`         |                       |

---

## Bibliografia Completa

### Pricing e Volatility Surface
- **Black, F., & Scholes, M.** (1973). The Pricing of Options and Corporate Liabilities.
  *Journal of Political Economy*, 81(3), 637–654.
- **Breeden, D. T., & Litzenberger, R. H.** (1978). Prices of State-Contingent Claims
  Implicit in Option Prices. *Journal of Business*, 51(4), 621–651.
- **Dupire, B.** (1994). Pricing with a Smile. *Risk*, 7(1), 18–20.
- **Fengler, M. R.** (2009). Arbitrage-Free Smoothing of the Implied Volatility Surface.
  *Quantitative Finance*, 9(4), 417–428.
- **Gatheral, J., & Jacquier, A.** (2014). Arbitrage-Free SVI Volatility Surfaces.
  *Quantitative Finance*, 14(1), 59–71.
- **Cont, R., & da Fonseca, J.** (2002). Dynamics of Implied Volatility Surfaces.
  *Quantitative Finance*, 2(1), 45–60.

### Variance Risk Premium e Domanda
- **Carr, P., & Wu, L.** (2009). Variance Risk Premiums.
  *Review of Financial Studies*, 22(3), 1311–1341.
- **Bakshi, G., & Kapadia, N.** (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium. *Review of Financial Studies*, 16(2), 527–566.
- **Garleanu, N., Pedersen, L. H., & Poteshman, A. M.** (2009). Demand-Based Option
  Pricing. *Review of Financial Studies*, 22(10), 4259–4299.

### Gaussian Process e Metodi Non-Parametrici
- **Rasmussen, C. E., & Williams, C. K. I.** (2006). *Gaussian Processes for Machine
  Learning.* MIT Press.
- **Cousin, A., Maatouk, H., & Rullière, D.** (2016). Kriging of Financial
  Term-Structures. *European Journal of Operational Research*, 255(2), 631–648.

### Bayesian Optimization
- **Snoek, J., Larochelle, H., & Adams, R. P.** (2012). Practical Bayesian Optimization
  of Machine Learning Algorithms. *NeurIPS*, 25, 2951–2959.
- **Frazier, P. I.** (2018). A Tutorial on Bayesian Optimization. *arXiv:1807.02811*.

### Backtest e Data Snooping
- **Bailey, D. H., & Lopez de Prado, M.** (2014). The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality.
  *Journal of Portfolio Management*, 40(5), 94–107.
- **White, H.** (2000). A Reality Check for Data Snooping.
  *Econometrica*, 68(5), 1097–1126.
- **Lopez de Prado, M.** (2018). *Advances in Financial Machine Learning.* Wiley.

---

*Documento di riferimento interno — GP Vol Surface Alpha Research Project*
