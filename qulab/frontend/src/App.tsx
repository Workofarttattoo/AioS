import { Link, Routes, Route, Navigate } from "react-router-dom";
import Teleport from "./routes/Teleport";
import Encoding from "./routes/Encoding";
import Governance from "./routes/Governance";
import FieldMaintenance from "./routes/FieldMaintenance";
import Help from "./widgets/Help";

export default function App(){
  return (
    <>
      <nav className="nav">
        <img src="/logo.svg" className="logo" alt="QuLab"/>
        <Link to="/teleport">Teleport</Link>
        <Link to="/encoding">Encoding</Link>
        <Link to="/governance">Governance</Link>
        <Link to="/field">Field</Link>
        <a href="http://localhost:8000/health" target="_blank">API</a>
        <Help/>
      </nav>
      <div className="container">
        <Routes>
          <Route path="/" element={<Navigate to="/teleport" />} />
          <Route path="/teleport" element={<Teleport/>} />
          <Route path="/encoding" element={<Encoding/>} />
          <Route path="/governance" element={<Governance/>} />
          <Route path="/field" element={<FieldMaintenance/>} />
        </Routes>
      </div>
    </>
  );
}
