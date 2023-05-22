import React, { useState, useEffect } from 'react' ;
import './Home.css';
import img from './image1.png'
import img2 from './image2.png'
import { List, Card, Button } from 'antd';
function Home() {

  return (
    <div className="div_container">
        <h1 className="homeHeader">
        <span style={{color:"#ca28a1"}}>Summ</span>
        <span style={{color:"#d2d2d2"}}>plify</span>
        </h1>
        <h2 className="homeHeader2">Condensing Complexity for Clarity</h2>
        <div className='div_button'>
        <a href='/main' className="a_Home" >Let's Start</a>
        </div>
        <div className="imgBg">
          <img src={img} className="img_Home"></img>
        </div>
    </div>
  );
}

export default Home;
