import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { JobsListPage } from "@/pages/JobsListPage";
import { UploadPage } from "@/pages/UploadPage";
import { JobPage } from "@/pages/JobPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<JobsListPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/jobs/:jobId" element={<JobPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
