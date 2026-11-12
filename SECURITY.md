# Security policy

## Supported versions

Only the latest released version receives fixes (ScientificLab is pre-1.0).

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's
private vulnerability reporting ("Security" tab -> "Report a vulnerability") on
the repository. We aim to acknowledge reports within a few days.

## Things to know

- Functions that accept equations or expressions as text (`solvers`,
  `mathematics.algebra`, `visualization.plot_function`) parse them with SymPy,
  which relies on `eval`. **Never pass untrusted text to them.**
- The graphical interface (`scientificlab-gui`) exposes those functions through a
  local web page. It listens on `127.0.0.1` only, requires a random per-session
  token and checks the `Host` and `Origin` headers. Do not put it behind a
  reverse proxy or bind it to another interface.
