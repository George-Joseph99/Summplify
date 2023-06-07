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
    const params = location.state;
    const received_data = params[1]
    const method = params[0]
    let isSimplifier = false;
    let isSummarizer = false;
    let isTranslator = false;

    if(method == 0){
      isSimplifier = true;
    }
    else if(method == 1){
      isSummarizer = true;
    }
    else if(method == 2){
      isTranslator = true;
    }
    let simplifierText = ''
    let summarizedText = ''
    let translatedText = ''
    let simplifierDefinitions = []
    if (isSimplifier) {
      console.log(received_data)
      simplifierText = received_data[0]
      simplifierDefinitions = received_data.slice(1)
    } else if(isSummarizer){
      summarizedText = received_data
    }
    else if(isTranslator){
      translatedText = received_data
    }

    if (isSimplifier)
    {
      const firstSpaceIndex = simplifierDefinitions[0].indexOf(" ");
      const secondSpaceIndex = simplifierDefinitions[0].indexOf(" ", firstSpaceIndex + 1);
      const thirdSpaceIndex = simplifierDefinitions[0].indexOf(" ", secondSpaceIndex + 1);
      const firstClosingBrackIndex = simplifierDefinitions[0].indexOf(")");
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
    <p className="p_output_summarizer">{summarizedText}</p>
    </div>) 
    : (<div></div>) }

    {isTranslator ? (<div> 
      <p className="p_output_summarizer">{translatedText}</p>
      </div>) 
      : (<div></div>)}
    </div>
  );
}

export default Output;
