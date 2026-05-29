# Track: Insurance AI
## Use Case Title: AI-Guided Conversion Flow — Der intelligente Conversion Coach

**Challenge Owner:** UNIQA
**Mentor(en):** TBD (Lumos + UNIQA Fachansprechpartner:in)
**Schwierigkeitsgrad:** Mittel bis Fortgeschritten
**Geschätzter Scope:** Ja, der Case ist in 36h realistisch umsetzbar, weil die fachliche Beratungslogik durch einen vorhandenen Chatbot abgedeckt wird und sich der Hackathon auf zwei klar abgegrenzte Bausteine konzentriert: die Interventionslogik des Conversion Coachs und die Persona-basierte Simulation auf dem Cluster.

---

### 🎯 Scope-Eingrenzung — BITTE ZUERST LESEN

Es gibt eine harte Scope-Grenze für diesen Track. Der Conversion Coach operiert **nur** auf der Strecke, die Nutzer:innen selbst online abschließen können:

| Im Scope ✅ | Außerhalb des Scope ❌ |
|---|---|
| **Privatarzt-Tarife** ("Bei Arztbesuchen" — Start & Optimal) | **Krankenhaus-Tarife** ("Im Krankenhaus" — Krankenhaus/Sonderklasse-Pfad) |
| **"Ich selbst"** — Versicherung nur für sich selbst | **"Andere Personen"** — Versicherung für andere (routet automatisch zum Berater) |
| **Online-abschließbare Tarife** (Start & Optimal) | **Beratungspflichtige Tarife** (Opt. Plus & Premium — routet zur Terminbuchung) |
| **Alle Infos der aktuellen Strecke müssen weiterhin abgefragt werden** — kein Schritt darf entfallen | Berater-Übergabe ist ein gültiger Exit, aber **kein Conversion-Erfolg** für diesen Track |
| Nutzer:innen, die online abschließen können → Coach hilft ihnen beim Abschluss | Nutzer:innen, die nicht online abschließen können → routen zum Berater (kein weiteres Coaching) |

**Conversion für diesen Track = Online-Abschluss (Start oder Optimal).** Alles, was zum Berater routet, ist außerhalb des Coaching-Scope — es ist ein sauberer Exit, kein Conversion-Gewinn.

Das heißt: Der Krankenhaus-Pfad (Step 5 Zusatzbausteine), der "andere Personen"-Zweig (Step 2) und die Tarife Opt. Plus/Premium sind **explizit aus dem aktiven Interventions-Scope ausgeschlossen**. Nutzer:innen, die diese Pfade wählen, werden zum Berater geroutet und verlassen den Funnel. Der Coach versucht nicht, sie zu retten — er hilft nur Nutzer:innen, die online abschließen *können*, auch tatsächlich abzuschließen.

Alle Informationen, die aktuell im Rechner abgefragt werden, müssen weiterhin erhoben werden — kein Schritt darf aus der In-Scope-Strecke entfernt werden.

---

### 1. Problem Statement (3–5 Sätze)

Kund:innen kommen mit konkretem Interesse an einer Krankenversicherung auf die UNIQA-Website, durchlaufen einen 15-stufigen Online-Rechner und brechen massenhaft ab — von 1.000 Personen, die starten, schließen am Ende nur etwa 56 online ab (5,6% Conversion Rate). Die Drop-off-Punkte sind bekannt (insbesondere die erstmalige Preisanzeige mit 66% Abbruch und der finale Preis mit 78% Abbruch), die *Gründe* dafür sind es nicht. Heute reagiert die Strecke nicht darauf, ob jemand zögert, vergleicht oder einfach den Preis abklärt — alle bekommen denselben statischen Funnel. Der Hackathon entwickelt einen **Conversion Coach**, der Unsicherheit und Abbruchintention in Echtzeit erkennt und im richtigen Moment passend interveniert, sowie ein **synthetisches Persona-Setup**, mit dem unterschiedliche Nutzerintentionen realistisch simuliert und Interventionen gegeneinander getestet werden können.

### 2. Warum das wichtig ist (Business Context)

- Die meisten Kund:innen, die abbrechen, waren eigentlich interessiert — sie brauchten nur im richtigen Moment die richtige Unterstützung. Ein Funnel, der sich an die Person anpasst (statt umgekehrt), hebt nicht nur die Conversion, sondern verbessert auch die Customer Experience an einer Stelle, die heute Frust erzeugt.
- Davon profitieren UNIQA als Versicherer (höhere Online-Conversion, bessere Lead-Qualität für die Beratungs-Übergabe) und perspektivisch alle Versicherer mit komplexen Online-Strecken — die hier entwickelten Logiken sind portabel.
- Der Bezug zu **European AI Sovereignty** liegt im Fokus auf eigenständig entwickelten Persona-Simulationen auf europäischer Compute-Infrastruktur, eigener Interventionslogik und einem testbaren System statt einer Black-Box-Lösung von der Stange. Es ist explizit *kein* LLM-Wrapper-Case: Die Intelligenz steckt in der Erkennungs- und Entscheidungslogik des Coachs sowie im Persona-Setup.

### 3. Erwartetes Outcome / Definition of Done

- **Minimum Viable Result:** Ein funktionierender Conversion-Coach-Prototyp (aufgesetzt auf einem vorhandenen versicherungsfachlichen Chatbot), der Verhaltenssignale erkennt und kontextbezogen interveniert, plus ein lauffähiges Persona-Simulationssetup mit mindestens drei Personas und einem dokumentierten Vorher-/Nachher-Vergleich der Conversion mit und ohne Coach.
- **Stretch Goals:** Persona-Varianten generieren (z.B. demographische Untervarianten je Segment), Interventions-Timing systematisch optimieren, Geschwindigkeit der Strecke als zusätzliche Test-Dimension, neue Abbruchmuster aufdecken, die in den heutigen UNIQA-Daten noch nicht sichtbar sind.
- **Lernziele für Teilnehmer:innen:** Aufbau einer regelbasierten oder lernenden Interventionslogik, Design synthetischer Personas mit realistischen Reaktionsmustern, großskaliges Journey-Simulieren auf Cluster-Infrastruktur, systematisches A/B-Testen von Logiken, sauberes Messen von Conversion-Effekten.
- **Format:** Lauffähiger Prototyp plus Simulationsbericht; optional ergänzt durch eine Live-Demo, in der eine Persona die Strecke durchläuft und der Coach an typischen Drop-off-Punkten eingreift.
- **Demonstrator Stretch Goal:** Eine Side-by-Side-Demo, die zwei identische Persona-Durchläufe nebeneinander zeigt — einmal mit, einmal ohne Coach — und sichtbar macht, an welcher Stelle die Intervention den Unterschied gemacht hat.

### 4. System-Spezifikation

- **Architektur:** Zweischichtig. Eine Erkennungs- und Entscheidungsschicht (der Coach) wird auf einen vorhandenen, versicherungsfachlichen Chatbot aufgesetzt. Der Chatbot beantwortet Fachfragen, der Coach entscheidet, *wann* und *wie* eingegriffen wird.
- **Fachlogik:** Wird als gegeben angenommen — der Chatbot kann Produktfragen, Begriffserklärungen und Tarifvergleiche bedienen. Diese Fähigkeit wird im Hackathon nicht neu gebaut.
- **Conversion Coach (Build-Fokus):**
  - Erkennt Verhaltenssignale wie Inaktivität, Zurücknavigieren, wiederholte Änderungen, ungewöhnlich lange Verweildauer, Hover-Patterns auf Preis-Elementen
  - Löst kontextbezogene Interventionen aus: vereinfachte Erklärungen, Vertrauenssignale, alternative Formulierungen, Markt-Vergleichshinweise — **Berater-Übergabe nur als Exit für Out-of-Scope-Pfade** (Krankenhaus, andere Personen, Opt. Plus/Premium), nicht als Coaching-Ziel
  - Ist so ausgelegt, dass unterschiedliche Persona-Typen mit verschiedenen Intentionen systematisch durchgespielt werden können
  - **Scope-Grenze:** Der Coach coacht nur die Privatarzt/"Ich selbst"/online-abschließbare Strecke. Nutzer:innen auf Krankenhaus-, "andere Personen"- oder Opt. Plus/Premium-Pfaden werden zum Berater geroutet — kein Coaching.
- **Persona-Setup:** Synthetische Personas mit unterschiedlichen Intentionen (Abschluss, Orientierung, Vergleich, Preis-Check) und Entscheidungslogiken. Die Personas sind nicht nur Testdaten, sondern zentrales Entwicklungsinstrument.
- **Trainingsmethoden:** Frei wählbar. Naheliegend sind regelbasierte Interventionslogiken, klassische ML-Klassifikatoren für Abbruchwahrscheinlichkeit, LLM-basierte Persona-Bots, RL für Interventions-Timing oder hybride Ansätze.
- **Constraints:** Reproduzierbarkeit auf Cluster-Infrastruktur, offene und nachvollziehbare Logik statt Black-Box-Empfehlungen. Es ist explizit *kein* reiner LLM-Wrapper-Case — die Coach-Logik muss eine eigenständige technische Komponente sein.

### 5. Aufgabenstruktur (Levels)

- **Level 1:** Die bestehende Strecke verstehen (Live-Rechner durchlaufen, Streckendoku lesen), die drei bereitgestellten Personas in lauffähige Persona-Bots überführen, eine erste Conversion-Coach-Logik mit klar definierten Trigger-Regeln und Interventionstypen bauen.
- **Level 2:** Die Logik im Persona-Setup testen, mindestens drei Interventionsvarianten gegeneinander vergleichen, Vorher-/Nachher-Conversion messen, erste Hypothesen formulieren, warum welche Intervention bei welcher Persona wirkt.
- **Level 3 / Stretch:** Großskalig auf dem Cluster simulieren (tausende Journeys, Persona-Varianten, Timing-Permutationen), neue Abbruchmuster aufdecken, die in den heutigen UNIQA-Daten heute nicht sichtbar sind, validierte Hypothesen für die Weiterentwicklung des Coachs in den Echtbetrieb formulieren.

### 6. Daten & Ressourcen

- **Personas:** Drei detaillierte Persona-Profile (Judith / Franz / Peter) liegen als Markdown-Briefings vor, ergänzt durch eine strukturierte `personas.json` mit quantitativen Werten aus der UNIQA-Segmentierungsforschung (Demographie, Versicherungsverhalten, Kaufkriterien, Kanalpräferenz pro CJ-Schritt, Lebensereignis-Trigger). Teams können diese Profile direkt als System-Prompts für Persona-Bots nutzen oder mit eigenen Varianten erweitern. **Hinweis zu Kanaldaten:** `personas.json` enthält `channel_preference_per_journey_step_pct_dominant_channel` — den präferierten Hauptkanal pro Schritt mit seinem Prozentwert (z.B. Beratung: via Berater 90%). Die vollständige 3-Kanal-Aufschlüsselung (Online / Berater / Kundenservice pro Schritt) ist nur für Segment 1 im Original-Segmentierungsbooklet verfügbar und wurde für diesen Datensatz nicht transkribiert. Teams, die Sekundärkanal-Verteilungen benötigen, sollten diese aus dem Dominanzwert und den Segment-Verhaltensmustern ableiten.
- **Segment-Verteilung im Online-Funnel:** Geschätzt 50% Segment 2 (Online-Affine), 30% Segment 1 (Aufsteigende Hybriden), 20% Segment 3 (Kundenservice-Affine). Quelle: UNIQA-interne Einschätzung, kein hartes Tracking.
- **Streckendokumentation:** Eine begleitende Markdown-Doku beschreibt die 15-stufige UNIQA-KV-Strecke mit den vier sichtbaren Phasen (Angaben → Produkt → Empfehlung → Abschluss), den bekannten Drop-off-Steps und den UI-Elementen. **Wichtig:** Nur die Privatarzt/"Ich selbst"/online-abschließbare Strecke ist Coaching-Scope. Der Krankenhaus-Pfad, der "andere Personen"-Zweig und die Tarife Opt. Plus/Premium führen automatisch in eine Terminbuchung und sind **außerhalb des aktiven Coaching-Scope**. Sie werden im Rechner weiterhin angeboten, der Coach versucht aber nicht, Nutzer:innen auf diesen Pfaden zu halten — sie werden zum Berater geroutet.
- **Produktdaten KV:** Vier Tarife (Start / Optimal / Opt. Plus / Premium) mit voraussichtlichen Prämien zwischen 38,74 € und 140,16 €/Monat, dazu Rückerstattungs-Höchstbeträge je Leistungsbereich (Arztleistungen, Medikamente, Therapeutische Behandlungen, Heilbehelfe, refraktive Augen-OP). **Im Scope für Coaching sind nur Start (€38,74) und Optimal (€68,14)** — die beiden online-abschließbaren Privatarzt-Tarife. Opt. Plus und Premium sind zur Simulation als Auswahloption relevant (Nutzer:innen können sie klicken → Coach routet zum Berater), aber nicht als Conversion-Ziel. Die Daten sind öffentlich auf [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) einsehbar und dürfen für die Simulation verwendet sowie synthetisch ergänzt werden.
- **Drop-off-Daten (real):** Aus der UNIQA-Funnel-Analyse (10.12.2025–01.02.2026):

  | Step | Was passiert | Drop-off |
  |---|---|---|
  | Tarifauswahl: Erste Preisanzeige | Initial Price | **66%** |
  | Auswahl Zusatzdeckungen | Additional Coverage | 24% |
  | Angabe Person: Finaler Preis | Final Price | **78%** |

  Daraus ergibt sich eine **aktuelle Online-Conversion-Baseline von ~5,6%** (von 1.000 Startern schließen ~56 ab). **Wichtig:** Alle Drop-off-Raten sind konditional auf das Erreichen des jeweiligen Schritts (d.h. relativ zur Kohorte, die an diesem Schritt angekommen ist, nicht zu den ursprünglichen 1.000 Startern). Die Survival-Rechnung: 1.000 Starter → 340 überleben Step 4 (34% überleben den 66% Drop-off) → 258 überleben Step 5 (76% überleben den 24% Drop-off) → ~57 überleben Step 7 (22% überleben den 78% Drop-off) ≈ 56 Abschlüsse = 5,6%.

- **Traffic-Quellen:** ~80% der Rechner-Zugriffe kommen über Paid Search und Organic Search, 70%+ zwischen 9 und 20 Uhr. Person:innen kommen also überwiegend mit konkretem Such-Intent in den Funnel.
- **Personabot "Team Tina":** Optionales Asset aus einem früheren UNIQA-Hackathon — Claude-basierter Persona-Bot auf Segmentierungsdaten. Zugang oder Datenexport ist bei UNIQA angefragt; falls nicht verfügbar, starten Teams mit den bereitgestellten Persona-Briefings.
- **Compute:** Training und Simulation auf dem Leonardo Cluster sind explizit vorgesehen. GPU-Quota pro Team: TBD.
- **NDAs / Datenschutz:** Alle bereitgestellten Daten sind freigegeben. Es werden keine personenbezogenen Echtdaten verwendet.

### 7. Evaluation & Benchmarking

- **Eval-Setup:** Jedes Team misst die Wirkung des Conversion Coachs gegen eine klar definierte Baseline (Strecke ohne Coach, gleiche Persona-Set, gleiche Anzahl Durchläufe). **Conversion für diesen Track = Online-Abschluss (Start oder Optimal).** Eine Berater-Übergabe ist ein gültiger Exit-Pfad für Out-of-Scope-Nutzer:innen (Krankenhaus, andere Personen, Opt. Plus/Premium), zählt aber **nicht** als Conversion-Erfolg für das Coaching-Ziel. Das Coaching-Ziel ist: Nutzer:innen, die online abschließen *können*, auch tatsächlich zum Abschluss bringen.
- **Drei zentrale Auswertungsdimensionen:**

  | # | Dimension | Was gemessen wird | Metrik(en) |
  |---|---|---|---|
  | 1 | **Conversion Uplift** | Wirkt der Coach? | Conversion Rate vs. Baseline, Drop-off-Reduktion je kritischem Step |
  | 2 | **Persona-Differenzierung** | Funktioniert er bei allen drei Personas? | Conversion pro Persona, Performance-Drop zwischen Personas |
  | 3 | **Interventions-Qualität** | Wann hilft er, wann stört er? | Trigger Precision/Recall, "Annoyance Rate" (unnötige Interventionen) |

- **Vergleichbarkeit:** Alle Teams arbeiten mit denselben drei Personas und derselben dokumentierten Strecke, sodass Ergebnisse direkt gegenübergestellt werden können. Eigene Persona-Varianten dürfen ergänzt werden, müssen dann aber im Bericht klar dokumentiert sein.
- **Visualisierung:** Erwartet werden mindestens Drop-off-Vergleiche (mit/ohne Coach pro Step), Conversion-Tabellen pro Persona und ein qualitatives Vorher-/Nachher-Beispiel einer kompletten Persona-Journey.
- **Test-Frequenz:** Sinnvoll sind iterative Simulationsläufe während der Entwicklung. Die konkrete Setup-Frequenz definieren die Teams selbst.

**Beispiel für Demonstrator-Outputs:**

- **Ohne Coach (Persona Franz, Segment 2 — Online-Affin):** Sieht erste Preisanzeige bei Tarifauswahl, vergleicht 4 Tarife, klickt einmal Premium ("nur nach Beratung"), klickt zurück, schließt den Tab → Abbruch.
- **Mit Coach (Persona Franz, gleiche Situation):** Coach erkennt das Rückwärtsnavigieren bei Premium-Tarif, blendet eine Erklärung ein ("Premium-Tarif erfordert kurzes Beratungsgespräch, Online-Abschluss ist mit Optimal jederzeit möglich"), zeigt einen Marktvergleich für Optimal → Franz wählt Optimal und konvertiert **online** → ✅ Conversion-Erfolg.
- **Ohne Coach (Persona Judith, Segment 1 — Aufsteigende Hybride):** Sieht Finalpreis nach Gesundheitsfragen, der höher ist als der initial gezeigte Preis, bricht ab.
- **Mit Coach (gleiche Situation):** Coach erkennt langes Verweilen auf der Finalpreis-Seite plus Hover auf "Abbrechen", erklärt transparent warum der Preis sich verändert hat, unterstützt Judith beim Online-Abschluss → ✅ Conversion-Erfolg.
- **Out-of-Scope-Beispiel (Persona wählt Krankenhaus oder andere Personen):** Coach erkennt die Pfadwahl, routet sauber zum Berater → kein Coaching, kein Conversion-Erfolg für diesen Track, aber korrekter Exit.

### 8. Technische Hinweise

- **Vorgeschlagener Tech Stack:** Python für Simulation und Eval, ein LLM-Framework freier Wahl (OpenAI/Anthropic/lokal) für die Persona-Bots, optional ein lightweightes Frontend (Streamlit, Gradio) für die Live-Demo. Persona-Bots brauchen keinen aufwändigen Stack — gute Prompts schlagen schwere Architektur.
- **Pflicht-Discovery (erste 30 Minuten):** Jedes Team durchläuft den echten Live-Rechner unter [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/) — aber **nur den Privatarzt/"Ich selbst"-Pfad**. Der Krankenhaus-Pfad und der "andere Personen"-Zweig sind außerhalb des Scope und müssen nicht im Detail durchlaufen werden.
- **Bekannte Stolpersteine:** Persona-Bots werden zu generisch wenn die Prompts zu kurz sind (nutzt die bereitgestellten Briefings vollständig), Interventions-Trigger feuern zu oft und werden nervig, die Versuchung ist groß zu viel auf einmal zu bauen statt eine Achse sauber zu validieren, faire Baseline-Vergleiche brauchen identische Persona-Seeds. **Scope-Stolperstein:** Nicht versuchen, den Krankenhaus-Pfad oder die "andere Personen"-Strecke zu coachen — diese sind außerhalb des Scope.
- **Bekannte Baseline:** ~5,6% Online-Conversion auf der echten Strecke; 66% Drop-off bei Erstpreis und 78% bei Finalpreis sind die zwei Ziele. Eine Coach-Logik, die diese Drop-offs sichtbar reduziert (auch nur bei einer Persona-Gruppe), ist bereits ein starkes Ergebnis.

### 9. Bewertungskriterien (track-spezifisch)

- Technische Tiefe der Conversion-Coach-Logik und nachvollziehbare Entscheidungsregeln
- Qualität und Realismus des Persona-Setups (eigene Personas oder Varianten sind willkommen)
- Aussagekraft des Baseline-vs.-Coach-Vergleichs auf der Simulation
- Reproduzierbarkeit und Klarheit der Evaluation (Persona-Seeds, Logging, Metrik-Definition)
- Belastbarkeit der Ableitungen — welche Hypothesen werden für den Echtbetrieb empfohlen und wie sind sie validiert?
- Qualität von Demo, Visualisierung und Ergebnisaufbereitung

### 10. Kontakt & Support während des Events

- **Challenge Owner:** Catarina, UNIQA (Slack-Channel `#help-insurance`)
- **Mentor vor Ort:** TBD
- **Fachlicher Ansprechpartner (per Slack/Telefon):** TBD UNIQA-seitig
- **Notfall-Kontakt:** Lumos-Desk in der Lobby