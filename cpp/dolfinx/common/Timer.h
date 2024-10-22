// Copyright (C) 2008 Anders Logg, 2015 Jan Blechta, 2024 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string>

namespace dolfinx::common
{

/// A timer can be used for timing tasks. The basic usage is
///
///   Timer timer("Assembling over cells");
///
/// The timer is started at construction and timing ends when the timer
/// is destroyed (goes out of scope). It is also possible to start and
/// stop a timer explicitly by
///
///   timer.start(); timer.stop();
///
/// Timings are stored globally and a summary may be printed by calling
///
///   list_timings();

template <typename chrono_timer = std::chrono::high_resolution_clock>
class Timer
{
public:
  /// Create timer
  ///
  /// If a task name is provided this enables logging to logger, otherwise (i.e.
  /// no task provided) nothing gets logged.
  Timer(std::optional<std::string> task = std::nullopt) : _task(task) {}

  /// Destructor
  ~Timer() = default;

  /// Zero and start timer
  void start() { _start_time = chrono_timer::now(); }

  /// Stop timer, return wall time elapsed and store timing data into
  /// logger
  template <typename unit = std::chrono::nanoseconds>
  double stop()
  {
    auto end_time = chrono_timer::now();
    auto elapsed = unit(end_time - _start_time).count();
    // TODO: reactivate
    // if (_task.has_value())
    //   TimeLogManager::logger().register_timing(_task.value(), wall, user,
    //   system);
    return elapsed;
  }

private:
  // Name of task
  std::optional<std::string> _task;

  chrono_timer::time_point _start_time;
};
} // namespace dolfinx::common
