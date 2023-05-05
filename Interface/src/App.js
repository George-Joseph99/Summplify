import './App.css';
import Home from './components/Home/Home'
import Main from './components/Main/Main'
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
       </Routes>
   </div> 
   </Router> 
  );
}

export default App;
