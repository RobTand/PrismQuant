"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  CHECK_NAME,
  evaluateClosingIssues,
  pullRequestIdentity,
  run,
} = require("./linked_issue.cjs");

const REPOSITORY = "RobTand/prismaquant";
const HEAD_SHA = "a".repeat(40);

function issue(number, repository = REPOSITORY) {
  return {
    __typename: "Issue",
    number,
    state: "OPEN",
    url: `https://github.com/${repository}/issues/${number}`,
    repository: { nameWithOwner: repository },
  };
}

function response(nodes, number = 253) {
  return {
    repository: {
      pullRequest: {
        number,
        closingIssuesReferences: { nodes },
      },
    },
  };
}

function context() {
  return {
    repo: { owner: "RobTand", repo: "prismaquant" },
    payload: {
      pull_request: {
        number: 253,
        base: { ref: "main", repo: { full_name: REPOSITORY } },
        head: { sha: HEAD_SHA },
      },
    },
    runId: 1234,
    serverUrl: "https://github.com",
  };
}

test("accepts a real same-repository issue", () => {
  assert.deepEqual(evaluateClosingIssues(response([issue(252)]), REPOSITORY, 253), {
    ok: true,
    issues: [
      {
        number: 252,
        state: "OPEN",
        url: "https://github.com/RobTand/prismaquant/issues/252",
      },
    ],
    totalClosingReferences: 1,
  });
});

test("rejects missing, cross-repository, and pull-request references", () => {
  const pullRequest = {
    ...issue(251),
    __typename: "PullRequest",
    url: "https://github.com/RobTand/prismaquant/pull/251",
  };
  const result = evaluateClosingIssues(
    response([issue(9, "RobTand/tessera"), pullRequest]),
    REPOSITORY,
    253,
  );
  assert.equal(result.ok, false);
  assert.deepEqual(result.issues, []);
  assert.equal(result.totalClosingReferences, 2);
  assert.equal(evaluateClosingIssues(response([]), REPOSITORY, 253).ok, false);
});

test("fails closed on malformed or mismatched API responses", () => {
  assert.throws(
    () => evaluateClosingIssues({}, REPOSITORY, 253),
    /GraphQL repository is missing or malformed/,
  );
  assert.throws(
    () => evaluateClosingIssues(response([issue(252)], 999), REPOSITORY, 253),
    /instead of #253/,
  );
  assert.throws(
    () =>
      evaluateClosingIssues(
        { repository: { pullRequest: { number: 253, closingIssuesReferences: {} } } },
        REPOSITORY,
        253,
      ),
    /nodes is missing or malformed/,
  );
});

test("takes the check SHA from the pull request head", () => {
  assert.deepEqual(pullRequestIdentity(context()), {
    owner: "RobTand",
    repo: "prismaquant",
    repository: REPOSITORY,
    pullNumber: 253,
    headSha: HEAD_SHA,
  });
});

test("an API error completes the actual-head check as failure", async () => {
  const statuses = [];
  const github = {
    graphql: async () => {
      throw new Error("GraphQL unavailable");
    },
    rest: {
      repos: {
        createCommitStatus: async (input) => statuses.push(input),
      },
    },
  };
  const failures = [];
  await run({
    github,
    context: context(),
    core: {
      error: () => {},
      info: () => {},
      setFailed: (message) => failures.push(message),
    },
  });

  assert.equal(statuses[0].sha, HEAD_SHA);
  assert.equal(statuses[0].context, CHECK_NAME);
  assert.equal(statuses[0].state, "pending");
  assert.equal(statuses.at(-1).sha, HEAD_SHA);
  assert.equal(statuses.at(-1).state, "failure");
  assert.match(failures.at(-1), /failed closed/);
});

test("a valid issue completes the actual-head check as success", async () => {
  const statuses = [];
  const github = {
    graphql: async () => response([issue(252)]),
    rest: {
      repos: {
        createCommitStatus: async (input) => statuses.push(input),
      },
    },
  };
  const failures = [];
  await run({
    github,
    context: context(),
    core: {
      error: () => {},
      info: () => {},
      setFailed: (message) => failures.push(message),
    },
  });

  assert.deepEqual(failures, []);
  assert.equal(statuses.at(-1).sha, HEAD_SHA);
  assert.equal(statuses.at(-1).context, CHECK_NAME);
  assert.equal(statuses.at(-1).state, "success");
  assert.match(statuses.at(-1).description, /#252/);
});
