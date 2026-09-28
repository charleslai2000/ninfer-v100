#pragma once
#include "artifact/v3_reader.h"
#include <iosfwd>
namespace ninfer::artifact::v3 { void inspect(const Reader& reader, std::ostream& out); }
