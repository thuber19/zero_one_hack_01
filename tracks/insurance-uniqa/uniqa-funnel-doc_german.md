# UNIQA Online-Rechner Krankenversicherung — Strecken-Dokumentation

**Quelle**: [uniqa.at/rechner/krankenversicherung](https://www.uniqa.at/rechner/krankenversicherung/)
**Stand**: Captured Mai 2026 (das Live-Verhalten kann sich ändern — Teams sollten zur Verifikation selbst durchlaufen)
**Drop-off Daten**: Zeitraum 10.12.2025 – 01.02.2026 (Quelle UNIQA Funnel-Analyse)

---

## 🎯 Scope-Eingrenzung — BITTE ZUERST LESEN

Bevor ihr die Streckendoku liest: Es gibt eine **harte Scope-Grenze** für diesen Hackathon-Track. Nicht alle Pfade im Rechner sind Coaching-Scope.

| Im Scope ✅ | Außerhalb des Scope ❌ |
|---|---|
| **Privatarzt-Tarife** ("Bei Arztbesuchen" — Start & Optimal) | **Krankenhaus-Tarife** ("Im Krankenhaus" — Sonderklasse-Pfad) |
| **"Ich selbst"** — Versicherung nur für sich selbst | **"Andere Personen"** — Versicherung für andere (routet zum Berater) |
| **Online-abschließbare Tarife** (Start & Optimal) | **Beratungspflichtige Tarife** (Opt. Plus & Premium — routet zur Terminbuchung) |
| **Alle Infos der aktuellen Strecke müssen weiterhin abgefragt werden** — kein Schritt darf entfallen | Berater-Übergabe ist ein gültiger Exit, aber **kein Conversion-Erfolg** für diesen Track |

**Conversion für diesen Track = Online-Abschluss (Start oder Optimal).** Alles, was zum Berater routet, ist außerhalb des Coaching-Scope — es ist ein sauberer Exit, kein Conversion-Gewinn.

Der Conversion Coach **coacht nur** Nutzer:innen, die online abschließen können (Privatarzt, "Ich selbst", Start/Optimal). Nutzer:innen auf Krankenhaus-Pfaden, "andere Personen"-Pfaden oder Opt. Plus/Premium werden sauber zum Berater geroutet — kein Coaching.

**Alle Informationen, die aktuell im Rechner abgefragt werden, müssen weiterhin erhoben werden.** Kein Schritt darf aus der In-Scope-Strecke entfernt werden. Der Coach darf nichts weglassen — er darf nur intervenieren, um den Abschluss zu unterstützen.

---

## Funnel-Übersicht

Die Strecke gliedert sich für die Person in vier sichtbare Phasen (Progress Bar):
**Angaben → Produkt → Empfehlung → Abschluss**

Aus dem Funnel-Tracking sind 15 Steps bekannt, von denen 4 die kritischen Drop-off-Punkte sind. Die wichtigsten Verzweigungen passieren früh — wer "im Krankenhaus" wählt, läuft in einen anderen Pfad als bei "bei Arztbesuchen".

Die Conversion-Logik des Rechners endet auf zwei Wegen:
1. **Online-Abschluss** möglich für die Tarife **Start** und **Optimal** (Privatarzt-Tarif "bei Arztbesuchen") — **dies ist der Coaching-Scope**
2. **Beratung erforderlich** für **Opt. Plus**, **Premium** sowie für alle Krankenhaus-Tarife und alle Konstellationen mit "andere Personen" → der Funnel mündet in eine Terminbuchung statt einem Online-Kauf — **dies ist außerhalb des Coaching-Scope**

**Für diesen Hackathon-Track ist nur Pfad 1 (Online-Abschluss) im Scope.** Pfad 2 routes zum Berater und wird vom Coach nicht weiter begleitet. Das bedeutet:
- Krankenhaus-Auswahl in Step 1 → sofort außerhalb des Scope, Coach routet zum Berater
- "Andere Personen" in Step 2 → sofort außerhalb des Scope, Coach routet zum Berater
- Opt. Plus/Premium in Step 4 → Coach erklärt, dass diese nur nach Beratung verfügbar sind, und leitet zum Berater weiter oder unterstützt bei der Auswahl von Start/Optimal

Alle Informationen, die aktuell im Rechner abgefragt werden, bleiben bestehen. Der Coach vereinfacht nicht die Datenabfrage — er unterstützt Nutzer:innen bei der Navigation und beim Abschluss.

---

## Die wichtigsten Steps im Detail

### Step 1 — Wo möchten Sie abgesichert sein?

**Phase**: Angaben
**Frage**: "Wo möchten Sie abgesichert sein?"
**UI**: Zwei große Cards, Mehrfachauswahl möglich
**Optionen**:
- **Bei Arztbesuchen** (Kassen-/Wahl-/Privatärzt:in, Schul- und Alternativmedizin, Telemedizin)
- **Im Krankenhaus** (Öffentliches Spital oder Privatklinik, Komfort im Zweibettzimmer, OP-Termin flexibel planen)

**Verzweigung**: Die Auswahl bestimmt komplett unterschiedliche Folgepfade. "Arztbesuche" führt in die Privatarzt-Tarif-Logik (4 Tarife mit Online-Abschluss-Option), "Krankenhaus" in den Sonderklasse-Pfad (komplexer, fast immer Beratung erforderlich).

**⚡ Scope-Hinweis**: Nur "Bei Arztbesuchen" ist im Coaching-Scope. Wer "Im Krankenhaus" wählt (oder beides), wird vom Coach zum Berater geroutet — kein weiteres Coaching. Alle Daten, die im aktuellen Rechner abgefragt werden, bleiben weiterhin vorhanden.

**UX-Beobachtung**: Keine Erklärung was die Konsequenz der Auswahl ist. Person:innen, die "alles" wollen, klicken vermutlich beide an — was den Pfad noch komplexer macht. Der Coach sollte hier transparent machen, dass der Krankenhaus-Pfad eine Beratung erfordert, während der Arztbesuch-Pfad online abschließbar ist.

---

### Step 2 — Für wen?

**Phase**: Angaben
**Frage**: "Wer soll versichert werden?"
**Optionen**:
- **Ich selbst** → Online-Abschluss möglich ✅ **In-Scope**
- **Andere Personen** → automatisch Beratungspfad ("Der Abschluss für andere Personen ist komplexer") ❌ **Out-of-Scope**

**Verzweigung**: "Andere Personen" beendet die Online-Strecke effektiv und routet direkt in die Terminbuchung.

**⚡ Scope-Hinweis**: Nur "Ich selbst" ist im Coaching-Scope. Der Coach muss erkennen, wenn jemand "andere Personen" wählt, und sauber zum Berater routen. Alle Informationen, die aktuell abgefragt werden, bleiben bestehen.

---

### Step 3 — Personendaten für Prämienschätzung

**Phase**: Angaben
**Frage**: "Um eine voraussichtliche individuelle Prämie für Sie zu berechnen, benötigen wir:"
**Pflichtfelder**:
- Geburtsdatum
- Sozialversicherung

**Kritischer Punkt**: Hier werden zum ersten Mal echte personenbezogene Daten abgefragt, bevor irgendein Preis gezeigt wurde. Das ist ein klassischer Vertrauens-Schwelle.

**⚡ Scope-Hinweis**: Dieser Schritt bleibt unverändert. Alle Daten müssen weiterhin abgefragt werden. Der Coach kann hier Vertrauen schaffen (z.B. Erklärung warum diese Daten nötig sind), aber keine Schritte entfernen.

---

### Step 4 — ⚠️ Tarifauswahl: Erste Preisanzeige (Drop-off 66%)

**Phase**: Produkt
**Frage**: "Welche Leistungen soll Ihre Privatarzt-Versicherung abdecken?"

**Hinweisbox** (oberhalb der Tarife):
> "Denken Sie an Ihren heutigen Bedarf, nicht an den in 20 Jahren. Nach 3 Jahren können Sie in einen anderen unserer vier Tarife wechseln, ohne erneute Gesundheitsprüfung!"

**UI**: Vergleichstabelle mit 4 Tarifen nebeneinander:

| Tarif         | Höchstbetrag/Jahr | Voraussichtliche Prämie | Status              |
| ------------- | ----------------- | ----------------------- | ------------------- |
| **Start**     | 1.400 EUR         | **38,74 EUR**           | Online abschließbar ✅ |
| **Optimal**   | 2.800 EUR         | **68,14 EUR**           | Online abschließbar ✅ |
| **Opt. Plus** | 4.200 EUR         | **96,66 EUR**           | Nur nach Beratung ❌ |
| **Premium**   | 8.400 EUR         | **140,16 EUR**          | Nur nach Beratung ❌ |

Aufgeschlüsselt nach Leistungsbereichen: Arztleistungen, Medikamente/Impfungen, Therapeutische Behandlungen, Heilbehelfe, refraktive Augen-OP.

**⚡ Scope-Hinweis**: Nur Start und Optimal sind Conversion-Ziele (online abschließbar). Opt. Plus und Premium sind im Rechner sichtbar, führen aber zum Berater. Der Coach sollte Nutzer:innen, die Opt. Plus/Premium anklicken, transparent erklären, dass diese eine Beratung erfordern, und sie bei der Auswahl von Start/Optimal unterstützen — statt sie zum Bleiben auf einem Beratung-erfordernden Pfad zu animieren.

**Warum so hoher Drop-off (66%)?** Mehrere plausible Gründe:
- Erste konkrete Zahl im Funnel — Preis-Schock
- Vier Optionen mit fünf verschiedenen Preis-Achsen = kognitive Überlast
- Die zwei attraktiveren Tarife (Opt. Plus, Premium) sind nur nach Beratung verfügbar → Frust bei Person:innen, die online abschließen wollten
- ROPO-Effekt: "Preis online angeschaut, kaufe ich später beim Berater" — nicht trackbar, aber laut UNIQA real
- Informationsbedarf zu unbekannten Begriffen ("refraktive Augen-OP", "Heilbehelfe")

**Conversion-Coach-Aufgabe**: Hier ist der **wichtigste Interventionsmoment**. Mögliche Hooks:
- Vergleich zum Markt aufzeigen ("Ihr Tarif ist günstiger als 80% der Privatarzt-Tarife")
- Begriffsboxen einblenden bei Hover/Klick
- Tarif-Empfehlung statt vollständiger Vergleichsmatrix für unsichere Person:innen
- Bei Klick auf Opt. Plus/Premium: transparent erklären, dass diese Beratung erfordern, und auf Start/Optimal als online-abschließbare Alternativen hinweisen
- "Was kostet das pro Tag?" — psychologische Umrechnung (€ 38,74/Monat = € 1,27/Tag)

---

### Step 5 — Auswahl Zusatzdeckungen (Drop-off 24%) — ❌ AUSSERHALB DES COACHING-SCOPE

**Phase**: Produkt (Krankenhaus-Pfad)
**Frage**: "Für welchen Versicherungsschutz interessieren Sie sich?"

**Optionen (Auswahl der vorhandenen)**:
- Sonderklasse nach Unfall
- Sonderklasse Select Kompakt
- Sonderklasse Select Optimal
- Sonderklassebehandlungen nach Unfall
- Sonderklassebehandlungen nach Unfall und schweren Erkrankungen
- Sonderklassebehandlungen für alle med. notwendigen Behandlungen mit Selbstbehalt
- Krankenhaus-Tagegeld
- Ersatz von Transportkosten
- Kinderbegleitkosten
- Ärztliche Zweitmeinung
- Psychologische Betreuung in Notfallsituationen
- Pauschale bei bösartigen Neubildungen (Krebs)
- Ambulante Diagnostik
- Hebamme (selbständig)

**Zusatzservices**:
- VitalPlan Vorsorge und Fitness
- Tagegeld

**⚡ Scope-Hinweis**: Dieser Schritt ist **nur relevant für den Krankenhaus-Pfad**, der außerhalb des Coaching-Scope liegt. Nutzer:innen, die "Bei Arztbesuchen" (Privatarzt) gewählt haben, gelangen nicht hierher. Nutzer:innen, die "Im Krankenhaus" gewählt haben, werden vom Coach zum Berater geroutet — kein weiteres Coaching auf diesem Pfad. Die Existenz dieses Schritts im Rechner bleibt unverändert (alle Infos bleiben abgefragt), aber der Coach interveniert hier nicht.

**UX-Beobachtung**: Bei "Krankenhaus" wird die Person mit ~15 möglichen Bausteinen konfrontiert, mit Fußnoten und Querverweisen. Niedrigerer Drop-off als bei Step 4 (24% vs 66%), aber wer hier ankommt, hat Step 4 schon überlebt und ist tendenziell entschlossener.

---

### Step 6 — Gesundheitsfragen

**Phase**: Angaben (Detailerhebung)
**Frage**: (Detail nicht final gecaptured — Teams sollten dies live verifizieren)

**Aus dem Briefing bekannt**: An dieser Stelle erhebt UNIQA die Gesundheitsdaten, die zur Berechnung der **finalen** Prämie nötig sind (vs. der "voraussichtlichen Prämie" aus Step 4).

**⚡ Scope-Hinweis**: Dieser Schritt bleibt unverändert. Alle Gesundheitsfragen müssen weiterhin beantwortet werden. Der Coach kann hier Vertrauen schaffen und Hilfestellung bieten, aber keine Fragen entfernen oder überspringen.

---

### Step 7 — ⚠️ Finaler Preis nach Personenangabe (Drop-off 78%)

**Phase**: Empfehlung
**Frage**: Finalisierte Prämie nach Gesundheitsprüfung
**Konsequenz**: Hier zeigt sich der echte, individualisierte Preis. Kann signifikant von der voraussichtlichen Prämie aus Step 4 abweichen.

**Warum noch höherer Drop-off (78%)?**
- Preis hat sich vermutlich verändert — meist nach oben (Risikoaufschlag)
- Wenn der finale Preis deutlich höher ist als die initiale Schätzung, fühlt sich die Person "vorgeführt"
- Vertrauensverlust: "warum stand vorher was anderes?"
- Bei Selbstbehalt-Optionen muss eine zusätzliche Entscheidung getroffen werden

**Conversion-Coach-Aufgabe**: Hier ist Schadensbegrenzung gefragt. Mögliche Hooks:
- Transparenz wieso sich der Preis verändert hat
- Alternative Tarif-Empfehlung wenn der finale Preis nicht passt (auf Start/Optimal fokussieren — diese sind online abschließbar)
- "Sie können trotzdem online abschließen" — viele wissen das nicht mehr
- Bei Opt. Plus/Premium-Wunsch: transparenter Hinweis, dass Beratung erforderlich ist, und Angebot für Start/Optimal als online-abschließbare Alternative

---

### Steps 8–11 — Beratungsanfrage-Pfad — ❌ AUSSERHALB DES COACHING-SCOPE

Wenn die Person in den Beratungspfad geroutet wird (Krankenhaus, andere Personen, Opt. Plus/Premium), folgen mehrere Steps:

**Step "Wo soll die Beratung stattfinden?"**
- Online-Videoberatung (NEU)
- Persönlich an einem UNIQA-Standort
- Per Telefon
- Persönlich zu Hause

**Step "Kundenstatus"**
- Neuer Kunde ohne Berater
- Bestandskunde, Online-Consulting-Team
- Bestandskunde, eigene:r Berater:in

**Step "Bundesland"** (Dropdown)

**Step "Service-Auswahl"** (welche Versicherungssparte)
- Health Insurance, Pension/Life, Household, Accident, Car, Leasing, Legal Protection, Travel, Leisure, Insurance Policy Review

**Step "Datumsauswahl"** (Kalender)

**Step "Terminvorschlag"**

**Step "Persönliche Daten"** (Name, Email, Telefon, Adresse, Geburtsdatum, Beruf, Sozialversicherung, Beratungsanliegen)

**Step "Summary & Bestätigung"**

**⚡ Scope-Hinweis**: Der Beratungsanfrage-Pfad ist **außerhalb des Coaching-Scope**. Der Coach routet Nutzer:innen sauber hierher, wenn sie Krankenhaus, andere Personen oder Opt. Plus/Premium gewählt haben, und begleitet sie nicht weiter. Die Steps bleiben im Rechner unverändert erhalten, werden aber vom Coach nicht aktiv unterstützt.

**Beobachtung**: Auch der Beratungs-Pfad ist 7+ Steps lang und hat mehrere Stellen, an denen Person:innen aussteigen könnten — der Funnel verlagert das Drop-off-Risiko, eliminiert es nicht. Für den Hackathon ist dies aber nicht Teil der Messung: Conversion = Online-Abschluss.

---

### Step 12+ — Abschluss (nur Tarife "Start" / "Optimal") — ✅ IN SCOPE

**Phase**: Abschluss
Die letzten Steps für Online-Abschluss decken vermutlich ab:
- Persönliche Daten (Name, Adresse, Kontakt)
- Versicherungsbeginn / Vertragslaufzeit
- Zahlungsdaten
- Einwilligungen (AGB, Datenschutz)
- Abschluss-Bestätigung

**⚡ Scope-Hinweis**: Dies ist der Zielbereich des Conversion Coachs — Nutzer:innen, die bis hier kommen, sollen den Online-Abschluss für Start oder Optimal erfolgreich abschließen.

**Diese Steps wurden in der Strecken-Begehung nicht vollständig durchlaufen** (würde echte Personendaten erfordern). Teams sollten dies bei Bedarf selbst verifizieren.

---

## Beobachtete Conversion-Killer (Hypothesen für Teams)

Die folgenden Hypothesen ergeben sich aus der Streckenstruktur und sollen den Teams als Ausgangspunkt dienen — nicht als Fakten:

1. **Preis-Schock bei erster Preisanzeige** (Step 4 → 66% weg)
2. **Beratung-Notwendigkeit bei den attraktivsten Tarifen** schafft Frust bei online-affinen Person:innen → Coach sollte hier transparent kommunizieren und auf Start/Optimal als online-abschließbare Alternativen hinweisen
3. **Gap zwischen voraussichtlicher und finaler Prämie** zerstört Vertrauen
4. **Kognitive Überlast** durch 4 Tarife × 6 Leistungskategorien × Fußnoten
5. **Fehlende Erklärung von Fachbegriffen** ("refraktive Augen-OP", "Selbstbehalt", "Sonderklasse")
6. **Keine Vergleichsmöglichkeit zum Markt** — Person:innen verlassen die Seite, um zu vergleichen, und kommen nicht zurück
7. **Sozialversicherungsnummer-Abfrage** als Vertrauensschwelle
8. **"Nur nach Beratung" als Sackgasse** für die Person:innen, die explizit online abschließen wollten → Coach sollte auf Start/Optimal als online-abschließbare Alternativen hinweisen, statt Nutzer:innen im Beratungslabyrinth zu begleiten