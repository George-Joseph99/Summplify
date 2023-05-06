import { Result } from 'antd';
import './App.css';
import Home from './components/Home/Home'
import Main from './components/Main/Main'
import Output from './components/Output/Output'

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
function App() {
  return (
    <Router>
    <div className="App">
       <Routes>
        <Route exact path="/" element={<Home />}>
        </Route>
        <Route exact path="/main" element={<Main />}>
        </Route>
        <Route exact path="/output" element={<Output />}>
        </Route>
       </Routes>
   </div> 
   </Router> 
  );
}

export default App;
