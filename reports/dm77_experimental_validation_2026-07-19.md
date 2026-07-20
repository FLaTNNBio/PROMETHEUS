# PROMETHEUS/DM77: report sperimentale del 19 luglio 2026

## Sintesi esecutiva

La pipeline software è completa e riproducibile nel perimetro sintetico: 111 test su
111 passano, la campagna prespecificata `full` contiene 102 run completi e la
validazione holdout aggiuntiva contiene 20 run completi. Tutti i modelli e i
campionatori usano seed espliciti.

La configurazione PROMETHEUS con il valore allocato medio più alto sui cinque nuovi
seed holdout è `holdout_dr_robust_cross`:

- stimatore del segnale: doubly robust (`dr`);
- aggregazione: mediana;
- winsorization: disattivata;
- pesatura per affidabilità: disattivata;
- accordo direzionale minimo: `0.60`;
- peso delle coppie cross-action (`beta_cross`): `1.00`.

Questa configurazione raggiunge un valore globale medio di **291,57** (IC95%
270,52–312,63), contro **287,62** (261,68–313,56) del default. Il confronto appaiato
sugli stessi cinque seed è **+3,95** (IC95% −9,30–17,20). Il vantaggio non è quindi
abbastanza preciso per giustificare la sostituzione del default.

Il risultato principale è negativo ma utile: nell'attuale esperimento sintetico a
1.200 pazienti, il ranker globale primario non dimostra superiorità rispetto ai
comparatori più forti. Nella validazione holdout della candidata, il ranking GBDT
diretto ottiene 297,87, il ranking per rischio 329,30 e l'oracolo sintetico 425,96.
PROMETHEUS supera in media il casuale, ma con incertezza ampia. La configurazione è
quindi una candidata di ricerca, non un nuovo default e non evidenza di efficacia
clinica.

Una successiva ablation full-scale su 10 seed ha confrontato il checkpoint corrente
`global_concordance` con `dr_policy_value`. Il checkpoint DR puro è stato rigettato:
non produce un vantaggio confermato, riduce in media il valore sintetico evaluation-only
di 93,07 e aumenta il tempo medio per run da circa 4,3 a 11,1 minuti. Il checkpoint
`global_concordance` resta quindi il default.

## Protocollo e tracciabilità

L'evidenza di questo report proviene esclusivamente da run effettivi:

| Blocco | Run | Seed | Scopo |
| --- | ---: | --- | --- |
| Ablation prespecificate `full` | 102 | 17, 37, 71 | stimatore, robustezza, coppie, falsificazioni e stress DGP |
| Validazione con configurazioni complete | 20 | 101, 131, 151, 181, 211 | confronto holdout di default e tre candidate |
| Checkpoint full-scale discovery + confirmation | 10 | 131, 149, 163, 179, 191, 211, 227, 239, 251, 269 | `global_concordance` contro `dr_policy_value` |
| Test software finali | 111 | espliciti dove stocastici | correttezza e contratti |

I seed holdout non compaiono nelle ablation usate per proporre le candidate. Il
criterio primario holdout è il valore globale allocato nel DGP
`baseline_identifiable`; la concordanza globale è un controllo di ranking. Gli
intervalli riportati riassumono la variabilità fra seed e, con cinque repliche, vanno
considerati ancora preliminari.

Configurazione del protocollo holdout:
[`configs/prometheus/dm77_holdout_configuration_suites.yaml`](../configs/prometheus/dm77_holdout_configuration_suites.yaml).

Stato della campagna full:
[`artifacts/dm77_method_ablations/master_status.json`](../artifacts/dm77_method_ablations/master_status.json).

Configurazione dell'ablation del checkpoint:
[`configs/prometheus/dm77_checkpoint_experiment_suites.yaml`](../configs/prometheus/dm77_checkpoint_experiment_suites.yaml).

## Come opera la pipeline

PROMETHEUS non stima una CATE individuale per poi ordinare i pazienti. Costruisce
direttamente un ordinamento globale delle opportunità paziente-azione:

1. usa solo caratteristiche misurate prima della data indice;
2. mantiene separati bisogno DM77, stato di cura corrente e azione causale;
3. stima nuisance model action-specific con cross-fitting;
4. costruisce segnali causali ripetuti e coppie within-action/cross-action;
5. apprende un punteggio ordinale globale, non un effetto calibrato individuale;
6. calibra separatamente sul validation set;
7. alloca azioni sotto budget, capacità, eligibility, supporto e vincoli clinico-operativi;
8. unisce `true_cate` e gli altri campi oracle soltanto dopo che ranking e allocazione
   sono stati fissati, per la valutazione sintetica.

Rischio e actionability causale restano output distinti. I comparator basati sul
rischio non diventano la pipeline primaria anche quando ottengono risultati sintetici
migliori.

## Risultati holdout delle configurazioni PROMETHEUS

| Configurazione | Valore globale, media (IC95%) | Concordanza globale, media (IC95%) | Efficienza oracle normalizzata | Benefit@budget |
| --- | ---: | ---: | ---: | ---: |
| Default | 287,62 (261,68–313,56) | 0,530 (0,466–0,595) | 0,115 | 5,115 |
| Outcome regression, coppie default | 289,60 (270,95–308,26) | 0,542 (0,480–0,604) | 0,214 | 4,919 |
| **DR robust cross** | **291,57 (270,52–312,63)** | 0,536 (0,473–0,599) | 0,141 | **5,167** |
| Outcome regression + mean robust cross | 274,63 (250,37–298,89) | **0,544 (0,484–0,604)** | 0,118 | 4,651 |

Differenze appaiate rispetto al default:

| Candidata | Differenza di valore (IC95%) | Differenza di concordanza (IC95%) |
| --- | ---: | ---: |
| Outcome regression, coppie default | +1,98 (−37,53–41,49) | +0,012 (0,000–0,024) |
| DR robust cross | +3,95 (−9,30–17,20) | +0,006 (−0,003–0,014) |
| Outcome regression + mean robust cross | −13,00 (−65,13–39,14) | +0,014 (0,004–0,024) |

Nessuna candidata domina contemporaneamente valore, concordanza e Benefit@budget.
La candidata DR robust cross è la migliore osservata sul criterio primario, ma il suo
intervallo appaiato include ampiamente zero.

## Confronto con i benchmark

La tabella usa i run holdout della candidata DR robust cross, quindi popolazioni e seed
sono condivisi fra i metodi.

| Metodo | Valore globale, media (IC95%) | Concordanza globale | Efficienza oracle | Benefit@budget |
| --- | ---: | ---: | ---: | ---: |
| Oracolo sintetico, solo valutazione | 425,96 (406,88–445,04) | 1,000 | 1,000 | 8,596 |
| Risk priority | 329,30 (286,03–372,58) | 0,694 | 0,389 | 7,117 |
| Cost-normalized risk priority | 319,28 (284,60–353,95) | 0,698 | 0,306 | 6,925 |
| Direct pairwise GBDT ranker | 297,87 (291,32–304,42) | 0,598 | 0,188 | 5,328 |
| **PROMETHEUS DR robust cross** | **291,57 (270,52–312,63)** | **0,536** | **0,141** | **5,167** |
| Random priority | 263,58 (232,93–294,23) | 0,501 | 0,000 | 4,396 |

Confronti appaiati del valore PROMETHEUS meno benchmark:

- contro direct pairwise GBDT: −6,30 (IC95% −37,65–25,06), vittoria in 1/5 seed;
- contro risk priority: −37,73 (−84,75–9,29), vittoria in 1/5 seed;
- contro cost-normalized risk: −27,70 (−63,32–7,91), vittoria in 1/5 seed;
- contro random: +27,99 (−21,97–77,95), vittoria in 3/5 seed.

Nel DGP sintetico base rischio e beneficio sono ancora fortemente allineati; il buon
risultato dei ranking di rischio non dimostra actionability causale e non autorizza a
fondere i due output. Il direct pairwise GBDT è invece un comparator di ranking diretto
che merita una verifica metodologica più ampia.

## Ablation full-scale del checkpoint

Il protocollo usa la configurazione contrastive-v2 a 20.000 pazienti, nuisance GBM
5-fold × 5 ripetizioni, 15.000 coppie within-action e 15.000 cross-action per epoca.
I primi cinque seed costituiscono la discovery; gli altri cinque erano separati prima
dell'esecuzione come confirmation. La decisione di avviare la conferma ha usato solo
il validation DR policy value, mai `true_cate` o oracle.

| Gruppo | Differenza validation DR | Differenza valore v2 evaluation-only | Differenza concordanza v2 |
| --- | ---: | ---: | ---: |
| Discovery, 5 seed | +0,370 (IC95% −0,431–1,171) | +18,00 (−61,90–97,90) | +0,000 (−0,013–0,013) |
| Confirmation, 5 seed | +0,179 (−0,318–0,676) | −204,13 (−770,90–362,63) | −0,023 (−0,086–0,040) |
| Tutti i 10 seed | +0,275 (−0,095–0,644) | −93,07 (−328,32–142,19) | −0,011 (−0,037–0,015) |

In sei seed i checkpoint scelgono esattamente la stessa epoca. Nel seed 239 il
checkpoint DR aumenta il valore DR di validazione di 0,896 ma riduce la concordanza
di valutazione da 0,747 a 0,634 e il valore sintetico di 1.020,67. Questo è compatibile
con overfitting alla stima DR di validazione. Il costo computazionale totale dei dieci
run è stato 111,1 minuti.

Il checkpoint `dr_policy_value` puro non viene adottato. Una futura alternativa
ammissibile è un criterio lessicografico che imponga prima una concordanza minima e
usi il policy value soltanto per distinguere checkpoint con ranking equivalente; va
comunque validato su seed nuovi.

### Segnale non-oracle sui benchmark full-scale

Le tabelle osservazionali held-out prodotte dai dieci nuovi run, senza target oracle,
indicano il direct pairwise GBDT come prossima candidata:

| Metodo | AUTOC DR held-out | QINI DR held-out | DR-OPE actionwise pesato |
| --- | ---: | ---: | ---: |
| Direct pairwise GBDT | 1,969 | 0,674 | 1,433 |
| DR random forest | 1,960 | 0,686 | 1,405 |
| Unified global ranker | 1,786 | 0,624 | non prodotto in questa tabella OPE |
| PROMETHEUS contrastive-v2 | 1,775 | 0,620 | 1,412 |
| Risk priority | 1,433 | 0,542 | 1,193 |

Nel confronto appaiato direct GBDT meno v2, la differenza AUTOC è +0,194
(IC95% −0,015–0,402; 7/10 vittorie), la differenza QINI è +0,055
(−0,010–0,119; 8/10 vittorie) e la differenza DR-OPE è +0,023
(−0,115–0,160; 6/10 vittorie). Il segnale è promettente ma non ancora conclusivo.

## Cosa insegnano le ablation di scoperta

### Segnale causale

Fra gli stimatori singolarmente variati, outcome regression ha la concordanza media
più alta (0,557) e l'efficienza media più alta (0,131), contro 0,548 e −0,106 del DR
default. Il valore medio passa da 264,26 a 275,28, ma Benefit@budget scende da 5,517 a
4,317. La differenza di valore cambia segno fra seed; il risultato non basta per
sostituire il DR.

Nell'ablation di robustezza, `median/no-winsor/no-reliability` massimizza il valore
medio osservato (289,86), mentre `mean/winsor/reliability` massimizza l'efficienza
normalizzata (0,095). Il default mantiene una concordanza leggermente superiore e un
Benefit@budget più alto. Anche qui le metriche non indicano un unico vincitore.

### Costruzione delle coppie

Il segnale più coerente riguarda le coppie cross-action. Con `beta_cross=0`, il valore
medio scende a circa 237,75–237,86 e la concordanza a 0,517–0,523. La combinazione
`min_direction_agreement=0.60`, `beta_cross=1.00` raggiunge 270,48 e 0,558. Il ranking
globale deve quindi conservare coppie cross-action quando outcome, orizzonte, unità e
direzione del beneficio sono semanticamente comparabili.

### Stato di cura e label permutate

Escludere lo stato di cura corrente riduce concordanza (0,548 → 0,530) e
Benefit@budget (5,517 → 4,350), anche se il valore medio aumenta da 264,26 a 269,87 per
la forte variabilità del seed 71. Lo stato di cura pre-index va mantenuto sia per
semantica sia per consistenza delle transizioni.

Permutare le label delle coppie riduce concordanza (0,548 → 0,528) e Benefit@budget
(5,517 → 4,843), ma non fa collassare il valore medio, che sale a 272,54 per effetto di
un singolo seed. Il controllo di falsificazione non è quindi completamente pulito e
richiede più repliche e un controllo più diretto del collegamento tra score,
calibrazione e allocatore.

## Stress d'identificazione

Questi risultati usano il default sui tre seed di scoperta. Il rango esclude soltanto
l'oracolo.

| Scenario | Valore PROMETHEUS | Rango non-oracle | Miglior metodo osservato | Valore migliore |
| --- | ---: | ---: | --- | ---: |
| Baseline identificabile | 264,26 | 12/14 | Cost-normalized risk | 337,03 |
| Strong observed confounding | 292,97 | 7/14 | Cost-normalized risk | 352,66 |
| Poor overlap | 285,21 | 7/14 | Cost-normalized risk | 328,64 |
| Risk-benefit misalignment | 198,34 | 8/14 | Unified local ranker | 225,47 |
| Targeted selection observed | 284,03 | 10/14 | Action mean priority | 340,08 |
| Hidden confounding | 277,58 | 7/14 | Risk priority | 341,70 |
| Combined stress | 234,05 | 3/14 | Direct pairwise GBDT | 252,74 |

Il ranker primario non è il migliore in nessuno dei sette scenari. Hidden confounding
e combined stress sono scenari di fallimento/sensibilità: non soddisfano le assunzioni
necessarie all'identificazione e non devono essere reinterpretati come test di
robustezza clinica.

## Controlli negativi

Nei DGP `null_treatment_effect` e `placebo_outcome`, tutte le metriche basate sul vero
effetto sintetico sono esattamente zero, come previsto dalla costruzione del DGP.
Tuttavia, per il PROMETHEUS default l'AUTOC osservazionale held-out sul controllo nullo è −1,667
(IC95% −1,910–−1,424); sul placebo è 0,826 (−0,694–2,346). Il primo risultato segnala
struttura spuriosa o instabilità nel diagnostico osservazionale e impedisce di
considerare la falsificazione completamente superata.

## Decisione e prossimi esperimenti

1. **Non modificare il default.** La candidata DR robust cross è il miglior risultato
   holdout smoke osservato per valore, ma il vantaggio è piccolo e compatibile con zero.
   Il checkpoint `dr_policy_value` full-scale è stato rigettato dopo la conferma.
2. **Conservare il ranking globale cross-action.** `beta_cross=0` degrada in modo
   consistente il comportamento nell'ablation.
3. **Mantenere separati rischio e actionability.** I benchmark di rischio sono forti
   nel DGP base, ma non sono stime causali né sostituti del ranker.
4. **Indagare la falsificazione nulla.** Prima di ulteriori ottimizzazioni va chiarito
   perché l'AUTOC held-out è sistematicamente negativo quando l'effetto è nullo.
5. **Separare ranking e allocazione nell'analisi.** Concordanza, Benefit@budget e valore
   allocato a volte si muovono in direzioni opposte; servono ablation della
   calibrazione e dell'allocatore con score fissati.
6. **Eseguire una conferma più grande.** Il campione smoke ha 1.200 pazienti e circa
   240 opportunità ordinate per run. Servono più repliche, campioni più grandi e una
   suite a `dataset_seed` fisso per misurare stabilità del ranking e dell'allocazione.
7. **Valutare il direct pairwise GBDT come alternativa di ranking diretto.** È il
   benchmark con il miglior AUTOC medio nei nuovi run full-scale, ma gli intervalli
   appaiati includono zero: serve una confirmation dedicata prima di renderlo primario.

## Limiti delle conclusioni

I risultati stabiliscono correttezza software e comportamento su popolazioni
interamente sintetiche sotto DGP dichiarati. Non stabiliscono efficacia clinica,
validità per la popolazione italiana, superiorità su ACG o readiness per il deployment.
Il punteggio PROMETHEUS resta ordinale e non è una stima calibrata dell'effetto
individuale. Non sono disponibili panel clinici DM77, coorti retrospettive target-trial,
validazione temporale/geografica o studi prospettici.

## Artefatti principali

- riepiloghi delle 102 ablation:
  [`artifacts/dm77_method_ablations_analysis`](../artifacts/dm77_method_ablations_analysis);
- run holdout completi:
  [`artifacts/dm77_holdout_configuration_validation`](../artifacts/dm77_holdout_configuration_validation);
- riepiloghi holdout per configurazione:
  [`artifacts/dm77_holdout_configuration_summary`](../artifacts/dm77_holdout_configuration_summary);
- run full-scale dell'ablation checkpoint:
  [`artifacts/dm77_checkpoint_experiments`](../artifacts/dm77_checkpoint_experiments);
- roadmap dei livelli di evidenza:
  [`docs/prometheus_validation_roadmap.md`](../docs/prometheus_validation_roadmap.md).
