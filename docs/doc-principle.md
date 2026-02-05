# Documentation Principle: Math Notes vs Code Docs

## Rule

Math lives in PDFs. Code docs live in mdBook. Never duplicate between them.

## PDFs (math/)

- Self-contained LaTeX documents
- Definitions, theorems, proofs
- Conceptual motivation and generalization
- No code, no API, no implementation details

## mdBook (book/)

- What the code does and how to use it
- API surface, function signatures, usage examples
- Benchmarks and performance tables
- Links to PDFs for underlying theory — one sentence of context, then the link

## Template for a module doc page

```markdown
# [Module Name]

[One sentence: what this module implements and why.]
Theory: [link to PDF](../../math/relevant-note.pdf).

## API

\`\`\`rust
// minimal working example
\`\`\`

## Parameters

Table listing all user-facing parameters with descriptions and typical values.
Exact columns depend on the module. Example:

\`\`\`markdown
| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| d         | Cyclotomic degree (power of 2) | 1024 |
| k         | Extension degree | 4 |
| q         | Modulus (odd prime) | 2^32 - 1 |
\`\`\`

## Benchmarks

Table comparing performance across relevant dimensions.
Must include: hardware, parameter config, and units. Example:

\`\`\`markdown
**Setup:** Apple M2, d=1024, k=4, single-threaded.

| m  | Naive (µs) | Tower (µs) | Speedup |
|----|-----------|------------|---------|
| 10 | 71.91     | 4.06       | 17.7×   |
| 11 | 295.24    | 9.38       | 31.5×   |
\`\`\`
```

## Benchmarks: where they go

- **PDFs**: benchmarks that validate a theoretical claim (e.g. "32× speedup from tower decomposition"). Include full setup: hardware, parameters, methodology. These are archival — they justify the math.
- **mdBook**: benchmarks tracking current implementation performance. These update with the code and may differ from PDF numbers as the implementation evolves.
- When both exist, mdBook links to the PDF for the original measurement and notes any divergence.

## Anti-patterns

- Restating theorems in markdown — reader gets a worse version of the PDF
- Code examples in PDFs — they go stale
- Benchmarks in PDFs without setup/config — not reproducible
- Long prose motivation in code docs — link the PDF instead