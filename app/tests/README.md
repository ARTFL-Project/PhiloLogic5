# PhiloLogic5 Frontend Tests

Unit tests for the Vue.js web application using [Vitest](https://vitest.dev/) + [Vue Test Utils](https://test-utils.vuejs.org/).

## Running tests

```bash
cd app/
npm run test          # single run
npm run test:watch    # re-runs on file changes
```

## Structure

```
tests/
├── fixtures/          # JSON responses from a real database (EEBO+ECCO)
│   ├── web_config.json
│   ├── concordance.json
│   ├── kwic.json
│   ├── collocation.json
│   ├── time_series.json
│   ├── aggregation.json
│   └── bibliography.json
├── helpers.js         # Shared test utilities (mock config, HTTP, router, mixins)
└── *.test.js          # Test files (one per component + mixins + store)
```

## Adding tests

1. If the component needs HTTP data, add a fixture to `fixtures/` or use `createMockHttp()` from `helpers.js`
2. Use `createGlobalConfig()` for simple components, or manually compose pinia/i18n/router for complex ones (see Aggregation or SearchArguments tests for examples)
3. Run `npm run test:watch` during development

## E2E-only gaps

TextNavigation's `handleScroll` (scroll-based nav bar visibility) and the DOM manipulation inside `toggleTableOfContents` after open (scrollIntoView, focus) require a real browser. Everything else in TextNavigation is unit-tested.
