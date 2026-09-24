#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

struct EdgeRecord {
  int u, v, label;
};

static bool operator<(const EdgeRecord &a, const EdgeRecord &b) {
  if (a.u != b.u) return a.u < b.u;
  if (a.v != b.v) return a.v < b.v;
  return a.label < b.label;
}

struct RunReader {
  ifstream file;
  EdgeRecord current{};
  bool valid = false;

  explicit RunReader(const string &path) : file(path, ios::binary) { next(); }

  void next() {
    valid = static_cast<bool>(
        file.read(reinterpret_cast<char *>(&current), sizeof(current)));
  }

  void close() {
    if (file.is_open()) file.close();
  }
};

struct RunItem {
  EdgeRecord edge;
  size_t run;
};

struct RunItemGreater {
  bool operator()(const RunItem &a, const RunItem &b) const {
    return b.edge < a.edge;
  }
};

static bool parseLine(const string &line, int &u, int &v, string &label) {
  if (line.empty()) return false;
  size_t p = 0;
  try {
    size_t q = 0;
    u = stoi(line, &q);
    p = q;
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
    v = stoi(line.substr(p), &q);
    p += q;
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
    if (p == line.size()) return false;
    size_t end = p;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t' &&
           line[end] != '\r') ++end;
    label.assign(line, p, end - p);
    return !label.empty();
  } catch (...) {
    return false;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " filename\n";
    return 1;
  }

  const string filename = argv[1];
  ifstream input(filename);
  if (!input) {
    cerr << "Failed to open the file " << filename << "\n";
    return 2;
  }

  unordered_map<int, int> vertices;
  unordered_map<string, int> labels;
  vertices.reserve(1 << 20);
  labels.reserve(1 << 16);
  vector<string> runPaths;
  vector<RunReader> runs;
  vector<EdgeRecord> chunk;
  constexpr size_t CHUNK_RECORDS = 1'000'000;

  auto cleanupTempFiles = [&]() {
    for (auto &run : runs) run.close();
    for (const string &path : runPaths) {
      if (remove(path.c_str()) != 0 && errno != ENOENT) {
        cerr << "Warning: failed to remove temporary file: " << path << '\n';
      }
    }
  };

  int nextVertex = 0, nextLabel = 0;
  uint64_t inputRecords = 0;

  auto flushChunk = [&]() -> bool {
    if (chunk.empty()) return true;
    sort(chunk.begin(), chunk.end());
    string path = filename + ".format_run_" + to_string(runPaths.size()) + ".tmp";
    ofstream out(path, ios::binary | ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(chunk.data()),
              static_cast<streamsize>(chunk.size() * sizeof(EdgeRecord)));
    if (!out) return false;
    out.close();
    runPaths.push_back(path);
    chunk.clear();
    chunk.shrink_to_fit();
    return true;
  };

  string line, label;
  int u, v;
  while (getline(input, line)) {
    if (!parseLine(line, u, v, label) || u == v) continue;
    auto [uit, uInserted] = vertices.emplace(u, nextVertex);
    if (uInserted) ++nextVertex;
    auto [vit, vInserted] = vertices.emplace(v, nextVertex);
    if (vInserted) ++nextVertex;
    auto [lit, lInserted] = labels.emplace(label, nextLabel);
    if (lInserted) ++nextLabel;
    chunk.push_back({uit->second, vit->second, lit->second});
    ++inputRecords;
    if (chunk.size() == CHUNK_RECORDS && !flushChunk()) {
      cerr << "Failed to write temporary run file\n";
      cleanupTempFiles();
      return 3;
    }
  }
  if (!flushChunk()) {
    cerr << "Failed to write temporary run file\n";
    cleanupTempFiles();
    return 3;
  }
  input.close();

  const string bodyPath = filename + ".format_body.tmp";
  ofstream body(bodyPath, ios::binary | ios::trunc);
  ofstream mapping(filename + "_mapping", ios::trunc);
  if (!body || !mapping) {
    cerr << "Failed to open output files\n";
    cleanupTempFiles();
    remove(bodyPath.c_str());
    return 4;
  }

  runs.reserve(runPaths.size());
  priority_queue<RunItem, vector<RunItem>, RunItemGreater> heap;
  for (size_t i = 0; i < runPaths.size(); ++i) {
    runs.emplace_back(runPaths[i]);
    if (runs.back().valid) heap.push({runs.back().current, i});
  }

  uint64_t labeledEdges = 0;
  uint64_t topologyEdges = 0;

  bool havePair = false;
  int currentU = -1, currentV = -1, lastLabel = -1;
  while (!heap.empty()) {
    RunItem item = heap.top();
    heap.pop();
    const EdgeRecord e = item.edge;
    if (!havePair || e.u != currentU || e.v != currentV) {
      if (havePair) body << '\n';
      body << e.u << ' ' << e.v << ' ' << e.label;
      currentU = e.u;
      currentV = e.v;
      lastLabel = e.label;
      havePair = true;
      ++topologyEdges;
      ++labeledEdges;
    } else if (e.label != lastLabel) {
      body << ',' << e.label;
      lastLabel = e.label;
      ++labeledEdges;
    }
    runs[item.run].next();
    if (runs[item.run].valid) heap.push({runs[item.run].current, item.run});
  }
  if (havePair) body << '\n';
  body.close();

  ofstream formatted(filename, ios::trunc);
  if (!formatted) {
    cerr << "Failed to open formatted output file\n";
    cleanupTempFiles();
    remove(bodyPath.c_str());
    return 5;
  }
  formatted << "vertices: " << nextVertex
            << " topologyEdges: " << topologyEdges
            << " labeledEdges: " << labeledEdges
            << " labels: " << nextLabel << '\n';
  ifstream bodyInput(bodyPath, ios::binary);
  formatted << bodyInput.rdbuf();
  bodyInput.close();
  formatted.close();

  mapping << "mapping relationships of vertices\n";
  vector<pair<int, int>> vertexMapping(vertices.begin(), vertices.end());
  sort(vertexMapping.begin(), vertexMapping.end());
  for (const auto &entry : vertexMapping)
    mapping << entry.first << " -> " << entry.second << '\n';
  mapping << "mapping relationships of labels\n";
  vector<pair<string, int>> labelMapping(labels.begin(), labels.end());
  sort(labelMapping.begin(), labelMapping.end());
  for (const auto &entry : labelMapping)
    mapping << entry.first << " -> " << entry.second << '\n';
  mapping.close();

  cleanupTempFiles();
  if (remove(bodyPath.c_str()) != 0) {
    cerr << "Warning: failed to remove temporary file: " << bodyPath << '\n';
  }
  cout << "Finished\n";
  return 0;
}
