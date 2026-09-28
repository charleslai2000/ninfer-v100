#include "artifact/v3_inspector.h"
#include <ostream>
namespace ninfer::artifact::v3 {
void inspect(const Reader& r,std::ostream& out){out<<"version: 3\n"<<"artifact_id: ";for(auto b:r.artifact_id())out<<std::hex<<unsigned(std::to_integer<unsigned char>(b));out<<std::dec<<"\n";auto& d=r.directory();out<<"files: "<<d.at("files").dump()<<"\ncomponents: "<<d.at("components").dump()<<"\nobjects: "<<d.at("objects").dump()<<"\nbindings: "<<d.at("bindings").dump()<<"\nuses: "<<d.at("uses").dump()<<"\nmetadata: "<<d.value("metadata",nlohmann::json::object()).dump()<<"\nprovenance: "<<d.value("provenance",nlohmann::json::object()).dump()<<"\n";}
}
