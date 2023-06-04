import React, { useState, useEffect } from 'react' ;
import axios from "axios";
import { GoCloudUpload } from "react-icons/go";
import { AiTwotoneFileAdd } from "react-icons/ai";
import { AiTwotoneDelete } from "react-icons/ai";
import './Output.css'
import { List, Card, Button } from 'antd';
import { useSearchParams } from 'react-router-dom';
import Search from 'antd/es/transfer/search';
import { useLocation } from 'react-router-dom';
import pic5 from './pic5.png'

function Output() {
    const [searchparams] = useSearchParams();
    const location = useLocation();
    const received_data = location.state;
    const method = location.method;
    let isSimplifier = false;
    let isSummarizer = false;
    console.log(method);

    const [fileUpload, setFileUpload] = useState(false);
    const [fileName, setFileName] = useState('');
    const [file, setFile] = useState('');
    var files = [];

    // useEffect(() => {
    //     axios.get('http://localhost:3000/output') // 1 will be changed to id
    //     .then((response) => {
    //     const data = response.data;
    //     setText(data['text']);

    //     }).catch(() => {
    //         alert('Error retrieving data');
    //     })
    //     }, []);

    // const handleTextInput = (e) => {
    //     console.log(e.target.value);
    //     setText(e.target.value);
    // }

    let textAndDetails = [];
    let simplifierText = ''
    let simplifierDefinitions = []
    if (Array.isArray(received_data)) {
      isSimplifier = true;
      textAndDetails = received_data;
      simplifierText = received_data[0]
      simplifierDefinitions = received_data.slice(1)
    } else {
      isSummarizer = true;
      textAndDetails = [received_data];
    }

    const firstSpaceIndex = simplifierDefinitions[0].indexOf(" ");
    const secondSpaceIndex = simplifierDefinitions[0].indexOf(" ", firstSpaceIndex + 1);
    const thirdSpaceIndex = simplifierDefinitions[0].indexOf(" ", secondSpaceIndex + 1);
    const firstClosingBrackIndex = simplifierDefinitions[0].indexOf(")");
    console.log(firstSpaceIndex)
    const handleDeleteIconClick = () => {
        setFileName('');
        setFileUpload(false);
        setFile('');
    }


  return (
    <div className="div_container">
    {isSimplifier ? (<div>
      <p className="p_output">{simplifierText}</p>
      <img src={pic5} className="img_Output"></img>
      <div className="div_definitions">
      {
      simplifierDefinitions.map((item, index) => (
        <p className="p_definition" key={index}><span style={{color:"#8d3b8d", fontWeight: "bold" }}>{item.slice(0,item.indexOf(" "))}</span>
        {item.slice(item.indexOf(" "), item.indexOf(" ",item.indexOf("(")+14))}
        <span style={{color:"green", fontWeight: "bold" }}>
        {item.slice(item.indexOf(" ",item.indexOf("(")+14), item.indexOf(")"))}
        </span>
        {item.slice(item.indexOf(")"))}
        </p>
      ))
    }
    </div>
    </div>) 
    : (<div></div>)}

    {isSummarizer ? (<div>
      {
      simplifierDefinitions.map((item, index) => (
        <p  key={index}>{item}</p>
      ))
    }
    </div>) 
    : (<div></div>) }
    </div>
  );
}

export default Output;
