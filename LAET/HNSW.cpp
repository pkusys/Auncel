/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <fstream>

#include "HNSW.h"
#include "AuxIndexStructures.h"

namespace faiss {

using idx_t = Index::idx_t;

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no + 1] -
    cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n)
{
  FAISS_THROW_IF_NOT(levels.size() == 0);
  int cur_n = nb_neighbors(level_no);
  for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
    cum_nneighbor_per_level[i] += n - cur_n;
  }
}

int HNSW::cum_nb_neighbors(int layer_no) const
{
  return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no,
                          size_t * begin, size_t * end) const
{
  size_t o = offsets[no];
  *begin = o + cum_nb_neighbors(layer_no);
  *end = o + cum_nb_neighbors(layer_no + 1);
}



HNSW::HNSW(int M) : rng(12345) {
  set_default_probas(M, 1.0 / log(M));
  max_level = -1;
  entry_point = -1;
  efSearch = 16;
  efConstruction = 500;
  upper_beam = 1;
  offsets.push_back(0);
}


int HNSW::random_level()
{
  double f = rng.rand_float();
  // could be a bit faster with bissection
  for (int level = 0; level < assign_probas.size(); level++) {
    if (f < assign_probas[level]) {
      return level;
    }
    f -= assign_probas[level];
  }
  // happens with exponentially low probability
  return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult)
{
  int nn = 0;
  cum_nneighbor_per_level.push_back (0);
  for (int level = 0; ;level++) {
    float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
    if (proba < 1e-9) break;
    assign_probas.push_back(proba);
    nn += level == 0 ? M * 2 : M;
    cum_nneighbor_per_level.push_back (nn);
  }
}

void HNSW::clear_neighbor_tables(int level)
{
  for (int i = 0; i < levels.size(); i++) {
    size_t begin, end;
    neighbor_range(i, level, &begin, &end);
    for (size_t j = begin; j < end; j++) {
      neighbors[j] = -1;
    }
  }
}


void HNSW::reset() {
  max_level = -1;
  entry_point = -1;
  offsets.clear();
  offsets.push_back(0);
  levels.clear();
  neighbors.clear();
}



void HNSW::print_neighbor_stats(int level) const
{
  FAISS_THROW_IF_NOT (level < cum_nneighbor_per_level.size());
  printf("stats on level %d, max %d neighbors per vertex:\n",
         level, nb_neighbors(level));
  size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
  for (int i = 0; i < levels.size(); i++) {
    if (levels[i] > level) {
      n_node++;
      size_t begin, end;
      neighbor_range(i, level, &begin, &end);
      std::unordered_set<int> neighset;
      for (size_t j = begin; j < end; j++) {
        if (neighbors [j] < 0) break;
        neighset.insert(neighbors[j]);
      }
      int n_neigh = neighset.size();
      int n_common = 0;
      int n_reciprocal = 0;
      for (size_t j = begin; j < end; j++) {
        storage_idx_t i2 = neighbors[j];
        if (i2 < 0) break;
        FAISS_ASSERT(i2 != i);
        size_t begin2, end2;
        neighbor_range(i2, level, &begin2, &end2);
        for (size_t j2 = begin2; j2 < end2; j2++) {
          storage_idx_t i3 = neighbors[j2];
          if (i3 < 0) break;
          if (i3 == i) {
            n_reciprocal++;
            continue;
          }
          if (neighset.count(i3)) {
            neighset.erase(i3);
            n_common++;
          }
        }
      }
      tot_neigh += n_neigh;
      tot_common += n_common;
      tot_reciprocal += n_reciprocal;
    }
  }
  float normalizer = n_node;
  printf("   nb of nodes at that level %ld\n", n_node);
  printf("   neighbors per node: %.2f (%ld)\n",
         tot_neigh / normalizer, tot_neigh);
  printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
  printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%ld)\n",
         tot_common / normalizer, tot_common);



}


void HNSW::fill_with_random_links(size_t n)
{
  int max_level = prepare_level_tab(n);
  RandomGenerator rng2(456);

  for (int level = max_level - 1; level >= 0; --level) {
    std::vector<int> elts;
    for (int i = 0; i < n; i++) {
      if (levels[i] > level) {
        elts.push_back(i);
      }
    }
    printf ("linking %ld elements in level %d\n",
            elts.size(), level);

    if (elts.size() == 1) continue;

    for (int ii = 0; ii < elts.size(); ii++) {
      int i = elts[ii];
      size_t begin, end;
      neighbor_range(i, 0, &begin, &end);
      for (size_t j = begin; j < end; j++) {
        int other = 0;
        do {
          other = elts[rng2.rand_int(elts.size())];
        } while(other == i);

        neighbors[j] = other;
      }
    }
  }
}


int HNSW::prepare_level_tab(size_t n, bool preset_levels)
{
  size_t n0 = offsets.size() - 1;

  if (preset_levels) {
    FAISS_ASSERT (n0 + n == levels.size());
  } else {
    FAISS_ASSERT (n0 == levels.size());
    for (int i = 0; i < n; i++) {
      int pt_level = random_level();
      levels.push_back(pt_level + 1);
    }
  }

  int max_level = 0;
  for (int i = 0; i < n; i++) {
    int pt_level = levels[i + n0] - 1;
    if (pt_level > max_level) max_level = pt_level;
    offsets.push_back(offsets.back() +
                      cum_nb_neighbors(pt_level + 1));
    neighbors.resize(offsets.back(), -1);
  }

  return max_level;
}


/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistFarther>& input,
  std::vector<NodeDistFarther>& output,
  int max_size)
{
  while (input.size() > 0) {
    NodeDistFarther v1 = input.top();
    input.pop();
    float dist_v1_q = v1.d;

    bool good = true;
    for (NodeDistFarther v2 : output) {
      float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

      if (dist_v1_v2 < dist_v1_q) {
        good = false;
        break;
      }
    }

    if (good) {
      output.push_back(v1);
      if (output.size() >= max_size) {
        return;
      }
    }
  }
}

// Load ground truth nearest neighbor database index
// for finding ground truth minimum termination condition.
void HNSW::load_gt(long label) {
  if (label == -1) {
    gtvector.clear();
  } else if (label == -2) {
    // For each query, create a separate vector to store the corresponding
    // ground truth nearest neighbor(s) indexes.
    std::vector<idx_t> newvector;
    gtvector.push_back(newvector);
  } else {
    gtvector[gtvector.size()-1].push_back(label);
  }
}

// Load the thresholds about when to make predictions.
// This is related to the choice of intermediate search result features.
void HNSW::load_thresh(long thresh) {
  if (thresh == -1) {
    pred_thresh.clear();
  } else {
    pred_thresh.push_back(thresh);
  }
}

// Load the prediction model.
void HNSW::load_model(char *file) {
  std::string filename(file);
  LightGBM::Boosting *booster =
    LightGBM::Boosting::CreateBoosting(std::string("gbdt"),
    filename.c_str());
  boosters.push_back(booster);
}

namespace {


using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;


/**************************************************************
 * Addition subroutines
 **************************************************************/


/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& resultSet1,
  int max_size)
{
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }

}


/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(HNSW& hnsw,
              DistanceComputer& qdis,
              storage_idx_t src, storage_idx_t dest,
              int level)
{
  size_t begin, end;
  hnsw.neighbor_range(src, level, &begin, &end);
  if (hnsw.neighbors[end - 1] == -1) {
    // there is enough room, find a slot to add it
    size_t i = end;
    while(i > begin) {
      if (hnsw.neighbors[i - 1] != -1) break;
      i--;
    }
    hnsw.neighbors[i] = dest;
    return;
  }

  // otherwise we let them fight out which to keep

  // copy to resultSet...
  std::priority_queue<NodeDistCloser> resultSet;
  resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
  for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
    storage_idx_t neigh = hnsw.neighbors[i];
    resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
  }

  shrink_neighbor_list(qdis, resultSet, end - begin);

  // ...and back
  size_t i = begin;
  while (resultSet.size()) {
    hnsw.neighbors[i++] = resultSet.top().id;
    resultSet.pop();
  }
  // they may have shrunk more than just by 1 element
  while(i < end) {
    hnsw.neighbors[i++] = -1;
  }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
  HNSW& hnsw,
  DistanceComputer& qdis,
  std::priority_queue<NodeDistCloser>& results,
  int entry_point,
  float d_entry_point,
  int level,
  VisitedTable &vt)
{
  // top is nearest candidate
  std::priority_queue<NodeDistFarther> candidates;

  NodeDistFarther ev(d_entry_point, entry_point);
  candidates.push(ev);
  results.emplace(d_entry_point, entry_point);
  vt.set(entry_point);

  while (!candidates.empty()) {
    // get nearest
    const NodeDistFarther &currEv = candidates.top();

    if (currEv.d > results.top().d) {
      break;
    }
    int currNode = currEv.id;
    candidates.pop();

    // loop over neighbors
    size_t begin, end;
    hnsw.neighbor_range(currNode, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t nodeId = hnsw.neighbors[i];
      if (nodeId < 0) break;
      if (vt.get(nodeId)) continue;
      vt.set(nodeId);

      float dis = qdis(nodeId);
      NodeDistFarther evE1(dis, nodeId);

      if (results.size() < hnsw.efConstruction ||
          results.top().d > dis) {

        results.emplace(dis, nodeId);
        candidates.emplace(dis, nodeId);
        if (results.size() > hnsw.efConstruction) {
          results.pop();
        }
      }
    }
  }
  vt.advance();
}


/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSW& hnsw,
                           DistanceComputer& qdis,
                           int level,
                           storage_idx_t& nearest,
                           float& d_nearest)
{
  for(;;) {
    storage_idx_t prev_nearest = nearest;

    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    for(size_t i = begin; i < end; i++) {
      storage_idx_t v = hnsw.neighbors[i];
      if (v < 0) break;
      float dis = qdis(v);
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    if (nearest == prev_nearest) {
      return;
    }
  }
}


}  // namespace


/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(DistanceComputer& ptdis,
                                   storage_idx_t pt_id,
                                   storage_idx_t nearest,
                                   float d_nearest,
                                   int level,
                                   omp_lock_t *locks,
                                   VisitedTable &vt)
{
  std::priority_queue<NodeDistCloser> link_targets;

  search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest,
                          level, vt);

  // but we can afford only this many neighbors
  int M = nb_neighbors(level);

  ::faiss::shrink_neighbor_list(ptdis, link_targets, M);

  while (!link_targets.empty()) {
    int other_id = link_targets.top().id;

    omp_set_lock(&locks[other_id]);
    add_link(*this, ptdis, other_id, pt_id, level);
    omp_unset_lock(&locks[other_id]);

    add_link(*this, ptdis, pt_id, other_id, level);

    link_targets.pop();
  }
}


/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(DistanceComputer& ptdis, int pt_level, int pt_id,
                          std::vector<omp_lock_t>& locks,
                          VisitedTable& vt)
{
  //  greedy search on upper levels

  storage_idx_t nearest;
#pragma omp critical
  {
    nearest = entry_point;

    if (nearest == -1) {
      max_level = pt_level;
      entry_point = pt_id;
    }
  }

  if (nearest < 0) {
    return;
  }

  omp_set_lock(&locks[pt_id]);

  int level = max_level; // level at which we start adding neighbors
  float d_nearest = ptdis(nearest);

  for(; level > pt_level; level--) {
    greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
  }

  for(; level >= 0; level--) {
    add_links_starting_from(ptdis, pt_id, nearest, d_nearest,
                            level, locks.data(), vt);
  }

  omp_unset_lock(&locks[pt_id]);

  if (pt_level > max_level) {
    max_level = pt_level;
    entry_point = pt_id;
  }
}


/** Do a BFS on the candidates list */

int HNSW::search_from_candidates(
  DistanceComputer& qdis, int k,
  idx_t *I, float *D,
  MinimaxHeap& candidates,
  VisitedTable& vt,
  int level, int nres_in) const
{
  int nres = nres_in;
  int ndis = 0;
  for (int i = 0; i < candidates.size(); i++) {
    idx_t v1 = candidates.ids[i];
    float d = candidates.dis[i];
    FAISS_ASSERT(v1 >= 0);
    if (nres < k) {
      faiss::maxheap_push(++nres, D, I, d, v1);
    } else if (d < D[0]) {
      faiss::maxheap_pop(nres--, D, I);
      faiss::maxheap_push(++nres, D, I, d, v1);
    }
    vt.set(v1);
  }

  bool do_dis_check = check_relative_distance;
  int nstep = 0;

  while (candidates.size() > 0) {
    float d0 = 0;
    int v0 = candidates.pop_min(&d0);

    if (do_dis_check) {
      // tricky stopping condition: there are more that ef
      // distances that are processed already that are smaller
      // than d0

      int n_dis_below = candidates.count_below(d0);
      if(n_dis_below >= efSearch) {
        break;
      }
    }

    size_t begin, end;
    neighbor_range(v0, level, &begin, &end);

    for (size_t j = begin; j < end; j++) {
      int v1 = neighbors[j];
      if (v1 < 0) break;
      if (vt.get(v1)) {
        continue;
      }
      vt.set(v1);
      ndis++;
      float d = qdis(v1);
      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d, v1);
      } else if (d < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d, v1);
      }
      candidates.push(v1, d);
    }

    nstep++;
    if (!do_dis_check && nstep > efSearch) {
      break;
    }
  }

  if (level == 0) {
#pragma omp critical
    {
      hnsw_stats.n1 ++;
      if (candidates.size() == 0) {
        hnsw_stats.n2 ++;
      }
      hnsw_stats.n3 += ndis;
    }
  }

  return nres;
}


/**************************************************************
 * Searching
 **************************************************************/

std::priority_queue<HNSW::Node> HNSW::search_from_candidate_unbounded(
  const Node& node,
  DistanceComputer& qdis,
  int ef,
  VisitedTable *vt) const
{
  int ndis = 0;
  std::priority_queue<Node> top_candidates;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

  top_candidates.push(node);
  candidates.push(node);

  vt->set(node.second);

  while (!candidates.empty()) {
    float d0;
    storage_idx_t v0;
    std::tie(d0, v0) = candidates.top();

    if (d0 > top_candidates.top().first) {
      break;
    }

    candidates.pop();

    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      int v1 = neighbors[j];

      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }

      vt->set(v1);

      float d1 = qdis(v1);
      ++ndis;

      if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
        candidates.emplace(d1, v1);
        top_candidates.emplace(d1, v1);

        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
      }
    }
  }

#pragma omp critical
  {
    ++hnsw_stats.n1;
    if (candidates.size() == 0) {
      ++hnsw_stats.n2;
    }
    hnsw_stats.n3 += ndis;
  }

  return top_candidates;
}

// For search_mode = 1.
// Generate training data for prediction-based approach.
void HNSW::search_from_candidate_unbounded_train(
  const Node& node,
  DistanceComputer& qdis,
  idx_t *I, float *D, int k,
  idx_t gt_idx,
  VisitedTable *vt) const
{
  int nres = 0; // number of valid results in the heap
  int thresh_idx = 0; // current pred_thresh timestamp
  idx_t ndis = 0; // number of distance evaluations
  // Number of distance evaluations when one of ground truth nearest
  // neighbor(s) is found. In other words the minimum termination condition.
  idx_t gt_dis = -1;
  // Current shortest distance between the query and found nearest neighbor(s).
  float best_D = node.first;
  // Found nearest neighbor(s) that have shortest distance to the query.
  std::vector<idx_t> best_I;
  best_I.push_back(node.second);
  // Distance(query, base layer start node), one of the features.
  float d_nearest = node.first;
  float d0, d1;
  float eps = 0.0000000001; // to avoid division by zero
  int feature_written [pred_thresh.size()];
  memset (feature_written, 0, pred_thresh.size()*sizeof(int));
  // 4 represents the number of intermediate search result features
  // at each pred_thresh timestamp.
  float * feature = new float [4*pred_thresh.size()];
  storage_idx_t v0, v1;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
  candidates.push(node);
  faiss::maxheap_push(++nres, D, I, node.first, node.second);
  vt->set(node.second);

  while (!candidates.empty()) {
    std::tie(d0, v0) = candidates.top();
    candidates.pop();
    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      v1 = neighbors[j];
      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }
      vt->set(v1);
      d1 = qdis(v1);
      ++ndis;

      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d1, v1);
      } else if (d1 < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d1, v1);
      }
      if (d1 < best_D) {
        // If the new search result is better than all previous results,
        // rebuild best_D and best_I.
        best_D = d1;
        best_I.clear();
        best_I.push_back(v1);
      } else if (d1 == best_D) {
        // If the new search result is as good as current best, update best_I.
        best_I.push_back(v1);
      }
      candidates.emplace(d1, v1);
    }
    if (thresh_idx < pred_thresh.size()) {
      // If the number of distance evaluations reach another pred_thresh
      // timestamp and the corresponding features were not written, record the
      // intermediate search result features.
      if (ndis >= pred_thresh[thresh_idx] && feature_written[thresh_idx] == 0) {
        faiss::maxheap_reorder (k, D, I);
        feature[thresh_idx*4] = D[0]; // top 1 intermediate search result
        feature[thresh_idx*4+1] = D[9]; // top 10 intermediate search result
        feature[thresh_idx*4+2] = D[0]/(d_nearest+eps);
        feature[thresh_idx*4+3] = D[9]/(d_nearest+eps);
        faiss::maxheap_heapify (k, D, I, D, I, k);
        feature_written[thresh_idx] = 1;
        thresh_idx++;
      }
    }
    if (gt_dis < 0) {
      // Check best_I to find if there is any ground truth nearest neighbor.
      // If so, update gt_dis to the current ndis.
      for (int i_best_I = 0; i_best_I < best_I.size(); i_best_I++) {
        for (int igt = 0; igt < gtvector[gt_idx].size(); igt++) {
          if (best_I[i_best_I] == gtvector[gt_idx][igt] && gt_dis < 0) {
            gt_dis = ndis;
            break;
          }
        }
        if (gt_dis >= 0) {
          break;
        }
      }
    }
    // If all the pred_thresh timestamps are satisfied and the ground truth
    // termination condition is found, stop searching.
    if (thresh_idx >= pred_thresh.size() && gt_dis >= 0) {
      break;
    }
  }
  // It's possible that there is not enough candidate nodes to meet some of
  // pred_thresh timestamps. In that case we just use the search results at
  // the end as the features for those timestamps.
  if (thresh_idx < pred_thresh.size()) {
    faiss::maxheap_reorder (k, D, I);
    for (int i_feature = thresh_idx; i_feature < pred_thresh.size(); i_feature++) { 
      feature[i_feature*4] = D[0];
      feature[i_feature*4+1] = D[9];
      feature[i_feature*4+2] = D[0]/(d_nearest+eps);
      feature[i_feature*4+3] = D[9]/(d_nearest+eps);
    }
    faiss::maxheap_heapify (k, D, I, D, I, k);
  }
  faiss::maxheap_reorder (k, D, I);
  // HACK: we overwrite the actual search result distances in D
  // by the features so that we can easily write the features by
  // reading the search results without additional APIs.
  D[0] = (float)gt_dis;
  D[1] = d_nearest;
  for (int i_feature = 0; i_feature < 4*pred_thresh.size(); i_feature++) {
    D[2+i_feature] = feature[i_feature];
  }
  delete [] feature;
}

// For search_mode = 2.
// Prediction-based adaptive learned early termination.
void HNSW::search_from_candidate_unbounded_pred(
  const Node& node,
  DistanceComputer& qdis,
  idx_t *I, float *D, int k,
  const float *x, size_t d, long pred_max,
  VisitedTable *vt) const
{
  int nres = 0; // number of valid results in the heap
  int thresh_idx = 0; // current pred_thresh timestamp
  idx_t ndis = 0; // number of distance evaluations
  // term_cond is the termination condition computed as
  // min((2**max(prediction,0)) * efSearch / 100.0, pred_max).
  // Here efSearch is used as a tunable multiplier.
  idx_t term_cond = -1;
  // Distance(query, base layer start node), one of the features.
  float d_nearest = node.first;
  float d0, d1;
  double eps = 0.0000000001; // to avoid division by zero
  storage_idx_t v0, v1;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
  candidates.push(node);
  faiss::maxheap_push(++nres, D, I, node.first, node.second);
  vt->set(node.second);

  while (!candidates.empty()) {
    std::tie(d0, v0) = candidates.top();
    candidates.pop();
    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      v1 = neighbors[j];
      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }
      vt->set(v1);
      d1 = qdis(v1);
      ++ndis;

      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d1, v1);
      } else if (d1 < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d1, v1);
      }
      candidates.emplace(d1, v1);
    }
    // If the termination condition is met, stop searching.
    if (term_cond > 0 && ndis >= term_cond) {
      break;
    }
    // If the number of distance evaluations reach another pred_thresh
    // timestamp, make a prediction.
    if (thresh_idx < pred_thresh.size() && ndis >= pred_thresh[thresh_idx]) {
      // double t0 = getmillisecs();
      double * input = new double[d+5];
      double * output = new double[1];
      for (idx_t i = 0; i < d; i++) {
        input[i] = (double)(x[i]); // the query vector
      }
      input[d] = (double)(d_nearest);
      faiss::maxheap_reorder (k, D, I);
      input[d+1] = D[0]; // top 1 intermediate search result
      input[d+2] = D[9]; // top 10 intermediate search result
      faiss::maxheap_heapify (k, D, I, D, I, k);
      input[d+3] = input[d+1]/(d_nearest+eps);
      input[d+4] = input[d+2]/(d_nearest+eps);
      // Make prediction.
      (boosters[thresh_idx])->PredictRaw(input, output, &tree_early_stop);
      term_cond = std::min(pred_max, (long)(ceil(pow(2.0,
        std::max((double)0,output[0]))*efSearch/100.0)));
      delete [] input;
      delete [] output;
      thresh_idx++;
      // printf("Done in %.3f ms\n", getmillisecs() - t0);
      if (ndis >= term_cond) {
        break;
      }
    }
  }
}

// For search_mode = 3.
// ndis-based fixed configuration to find the minimum number
// of distance evaluations to reach certain accuracy targets. This is
// needed for generating training data and grid search on different
// intermediate search result features.
void HNSW::search_from_candidate_unbounded_ndis(
  const Node& node,
  DistanceComputer& qdis,
  idx_t *I, float *D,
  int k, VisitedTable *vt) const
{
  int nres = 0;
  idx_t ndis = 0;
  float d0, d1;
  storage_idx_t v0, v1;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;
  candidates.push(node);
  faiss::maxheap_push(++nres, D, I, node.first, node.second);
  vt->set(node.second);
  while (!candidates.empty()) {
    std::tie(d0, v0) = candidates.top();
    candidates.pop();
    size_t begin, end;
    neighbor_range(v0, 0, &begin, &end);

    for (size_t j = begin; j < end; ++j) {
      v1 = neighbors[j];
      if (v1 < 0) {
        break;
      }
      if (vt->get(v1)) {
        continue;
      }
      vt->set(v1);
      d1 = qdis(v1);
      ++ndis;

      if (nres < k) {
        faiss::maxheap_push(++nres, D, I, d1, v1);
      } else if (d1 < D[0]) {
        faiss::maxheap_pop(nres--, D, I);
        faiss::maxheap_push(++nres, D, I, d1, v1);
      }
      candidates.emplace(d1, v1);
    }
    // Here we use efSearch as the threshold to stop searching when the number
    // of distance evaluations reach a fixed configuration.
    if (ndis >= efSearch) {
      break;
    }
  }
}

void HNSW::search(DistanceComputer& qdis, int k,
                  idx_t *I, float *D,
                  VisitedTable& vt) const
{
  if (upper_beam == 1) {

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    for(int level = max_level; level >= 1; level--) {
      greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
    }

    int ef = std::max(efSearch, k);
    if (search_bounded_queue) {
      MinimaxHeap candidates(ef);

      candidates.push(nearest, d_nearest);

      search_from_candidates(qdis, k, I, D, candidates, vt, 0);
    } else {
      std::priority_queue<Node> top_candidates =
        search_from_candidate_unbounded(Node(d_nearest, nearest),
                                        qdis, ef, &vt);

      while (top_candidates.size() > k) {
        top_candidates.pop();
      }

      int nres = 0;
      while (!top_candidates.empty()) {
        float d;
        storage_idx_t label;
        std::tie(d, label) = top_candidates.top();
        faiss::maxheap_push(++nres, D, I, d, label);
        top_candidates.pop();
      }
    }

    vt.advance();

  } else {
    int candidates_size = upper_beam;
    MinimaxHeap candidates(candidates_size);

    std::vector<idx_t> I_to_next(candidates_size);
    std::vector<float> D_to_next(candidates_size);

    int nres = 1;
    I_to_next[0] = entry_point;
    D_to_next[0] = qdis(entry_point);

    for(int level = max_level; level >= 0; level--) {

      // copy I, D -> candidates

      candidates.clear();

      for (int i = 0; i < nres; i++) {
        candidates.push(I_to_next[i], D_to_next[i]);
      }

      if (level == 0) {
        nres = search_from_candidates(qdis, k, I, D, candidates, vt, 0);
      } else  {
        nres = search_from_candidates(
          qdis, candidates_size,
          I_to_next.data(), D_to_next.data(),
          candidates, vt, level
        );
      }
      vt.advance();
    }
  }
}

// Customized search() for search_mode = 1, 2, 3.
void HNSW::search_custom(DistanceComputer& qdis, int k,
                         idx_t *I, float *D,
                         int search_mode, idx_t gt_idx,
                         const float *x, size_t d, long pred_max,
                         VisitedTable& vt) const
{
  //  greedy search on upper levels
  storage_idx_t nearest = entry_point;
  float d_nearest = qdis(nearest);

  for(int level = max_level; level >= 1; level--) {
    greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
  }
  if (search_mode == 1) {
    search_from_candidate_unbounded_train(Node(d_nearest, nearest),
      qdis, I, D, k, gt_idx, &vt);
  } else if (search_mode == 2) {
    search_from_candidate_unbounded_pred(Node(d_nearest, nearest),
      qdis, I, D, k, x, d, pred_max, &vt);
  } else if (search_mode == 3) {
    search_from_candidate_unbounded_ndis(Node(d_nearest, nearest),
      qdis, I, D, k, &vt);
  }
  vt.advance();
}

void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
  if (k == n) {
    if (v >= dis[0]) return;
    faiss::heap_pop<HC> (k--, dis.data(), ids.data());
    --nvalid;
  }
  faiss::heap_push<HC> (++k, dis.data(), ids.data(), v, i);
  ++nvalid;
}

float HNSW::MinimaxHeap::max() const {
  return dis[0];
}

int HNSW::MinimaxHeap::size() const {
  return nvalid;
}

void HNSW::MinimaxHeap::clear() {
  nvalid = k = 0;
}

int HNSW::MinimaxHeap::pop_min(float *vmin_out) {
  assert(k > 0);
  // returns min. This is an O(n) operation
  int i = k - 1;
  while (i >= 0) {
    if (ids[i] != -1) break;
    i--;
  }
  if (i == -1) return -1;
  int imin = i;
  float vmin = dis[i];
  i--;
  while(i >= 0) {
    if (ids[i] != -1 && dis[i] < vmin) {
      vmin = dis[i];
      imin = i;
    }
    i--;
  }
  if (vmin_out) *vmin_out = vmin;
  int ret = ids[imin];
  ids[imin] = -1;
  --nvalid;

  return ret;
}

int HNSW::MinimaxHeap::count_below(float thresh) {
  int n_below = 0;
  for(int i = 0; i < k; i++) {
    if (dis[i] < thresh) {
      n_below++;
    }
  }

  return n_below;
}


}  // namespace faiss
