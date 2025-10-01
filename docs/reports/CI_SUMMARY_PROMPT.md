# CI/CD Summary Request Prompt

Use this prompt with any AI coding agent to get a comprehensive summary of the current Continuous Integration and Continuous Deployment setup for this repository.

---

## Prompt for AI Agent

```
Please analyze the current CI/CD infrastructure for this repository and provide a comprehensive summary including:

1. **CI Pipeline Overview**
   - What CI/CD platforms are configured? (GitHub Actions, GitLab CI, Jenkins, CircleCI, etc.)
   - Where are the configuration files located?
   - What triggers are set up? (push, PR, tags, scheduled, manual)

2. **Pipeline Jobs & Stages**
   - List all jobs/stages in the pipeline
   - What does each job do? (test, lint, build, deploy, etc.)
   - What are the dependencies between jobs?
   - Are jobs running in parallel or sequentially?

3. **Test Execution**
   - Which test frameworks are being used?
   - What test commands are executed?
   - Is code coverage measured? What's the current coverage target?
   - Are there different test suites? (unit, integration, e2e)
   - What's the typical test execution time?

4. **Environment & Matrix**
   - What Python/language versions are tested?
   - What operating systems are tested? (Linux, Windows, macOS)
   - Are there different dependency configurations tested?
   - What's the matrix strategy?

5. **Dependency Management**
   - How are dependencies installed? (pip, uv, poetry, etc.)
   - Is there dependency caching configured?
   - Are dependencies locked/pinned?
   - Is there a dependency security scanning step?

6. **Code Quality Checks**
   - What linters are configured? (ruff, flake8, pylint, etc.)
   - What formatters are checked? (black, isort, autopep8, etc.)
   - Are there type checking tools? (mypy, pyright, etc.)
   - Is there any static analysis? (sonarqube, codeclimate, etc.)

7. **Pre-commit Hooks**
   - Are pre-commit hooks configured?
   - What hooks are enabled?
   - Are pre-commit checks also run in CI?
   - Is there hook validation on PRs?

8. **Build & Artifacts**
   - Are there build steps? (Docker images, packages, binaries)
   - Are build artifacts stored/uploaded?
   - Where are artifacts published? (PyPI, Docker Hub, GitHub Releases)
   - Is there versioning/tagging automation?

9. **Deployment**
   - Are there deployment steps configured?
   - What environments are deployed to? (dev, staging, prod)
   - What deployment strategy is used? (blue-green, rolling, canary)
   - Are there smoke tests after deployment?

10. **Status & Health**
    - What's the current CI status? (passing, failing, pending)
    - When was the last successful run?
    - Are there any known CI issues or flaky tests?
    - What's the CI success rate over the last 30 days?

11. **Performance & Optimization**
    - What's the average pipeline duration?
    - Are there caching mechanisms in place?
    - Are jobs optimized for speed?
    - What's the bottleneck in the pipeline?

12. **Security & Compliance**
    - Are there security scanning tools? (bandit, safety, snyk)
    - Is there secrets scanning?
    - Are there compliance checks?
    - Is there license compliance checking?

13. **Notifications & Reporting**
    - Where do build notifications go? (email, Slack, Discord)
    - Are there status badges in README?
    - Is there integration with code coverage services? (Codecov, Coveralls)
    - Are there PR comments with test results?

14. **Gaps & Recommendations**
    - What's missing from the current CI setup?
    - What could be improved?
    - Are there best practices not being followed?
    - What are the next steps to enhance CI/CD?

Please format the response as a markdown document with clear sections, code examples where relevant, and actionable recommendations. Include any warnings about deprecated configurations or potential issues.
```

---

## Expected Output Format

The AI should provide a structured markdown report with:

- Clear section headers matching the questions above
- Code snippets showing relevant configuration
- Status indicators (✅ Configured, ⚠️ Needs Attention, ❌ Missing)
- Specific file paths and line numbers for references
- Actionable recommendations with priority levels
- Comparison with industry best practices

---

## Usage Examples

### Example 1: Quick Status Check
```bash
# For a quick check, you can simplify the prompt:
"Summarize the CI/CD setup: what platforms, what tests, current status?"
```

### Example 2: Deep Dive on Testing
```bash
# For detailed test analysis:
"Focus on questions 3, 10, and 11 from the CI summary prompt - analyze test execution, status, and performance"
```

### Example 3: Security Audit
```bash
# For security-focused review:
"Focus on question 12 from the CI summary prompt - provide a security audit of the CI/CD pipeline"
```

---

## Context to Provide

When using this prompt, also provide:

1. **Repository Structure:**
   - `.github/workflows/` directory contents
   - `.gitlab-ci.yml` or other CI config files
   - `pyproject.toml` or `setup.py` for dependency info
   - `.pre-commit-config.yaml` if exists

2. **Recent CI History:**
   - Last 5-10 CI run results
   - Any recent CI-related issues or PRs
   - Current branch status

3. **Project Context:**
   - Main programming language
   - Project type (library, application, service)
   - Deployment targets (if any)

---

## Quick Reference: Current Repository

Based on the recent PR merges, this repository has:

- **CI Platform:** GitHub Actions (`.github/workflows/ci.yml`)
- **Pre-commit:** Configured (`.pre-commit-config.yaml`)
- **Test Framework:** pytest
- **Python Versions:** 3.10, 3.11, 3.12
- **Dependency Tool (local):** UV
- **Dependency Tool (CI):** pip ⚠️ (inconsistency)

Run the full prompt above to get complete details!

---

## Notes

- This prompt is designed to work with any AI coding assistant (GitHub Copilot, Claude, GPT-4, etc.)
- Adjust the prompt based on your specific repository's CI complexity
- For multi-repo projects, run this prompt for each repository
- Update this prompt template as CI/CD practices evolve

---

**Last Updated:** October 1, 2025  
**Maintained By:** Development Team
