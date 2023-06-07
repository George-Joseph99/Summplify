import React, { useState, useEffect } from 'react' ;
import axios from "axios";
import { GoCloudUpload } from "react-icons/go";
import { AiTwotoneFileAdd } from "react-icons/ai";
import { AiTwotoneDelete } from "react-icons/ai";
import {TfiWrite} from "react-icons/tfi";
import './Main.css';
import { List, Card, Button } from 'antd';
import { createSearchParams, useNavigate } from 'react-router-dom';
import { useHistory } from 'react-router-dom';

function Main() {

    const [fileUpload, setFileUpload] = useState(false);
    const [fileName, setFileName] = useState('');
    const [fileData, setFileData] = useState('');
    const [file, setFile] = useState('');
    const [text, setText] = useState('');
    const [isHovering, setIsHovering] = useState(false);
    const [isHoveringSimp, setIsHoveringSimp] = useState(false);
    const [summIsClicked, setSummIsClicked] = useState(false);
    const [simIsClicked, setSimIsClicked] = useState(false);
    const [tranIsClicked, setTranIsClicked] = useState(false);
    const [isHoveringAgain, setIsHoveringAgain] = useState(false);
    const [extractiveOrAbstractive, setExtractiveOrAbstractive] = useState('1');
    const [showDefinitions, setShowDefinitions] = useState('0');
    const [sliderValue, setSliderValue] = useState(3);
    var files = [];
    const navigate = useNavigate();

    const handleGoButtonClick = () => {
        console.log(text)
        if(simIsClicked){
            const dataToSimplify = {
                "text": text,
                "summarizeOrSimplify": 0,   // 2 for translate, 1 for summary, 0 for simplify
                "showDefinitions": showDefinitions
            }
            axios.post('http://localhost:8000/main',  dataToSimplify)
            .then(response => {
            console.log("rr", response.data);
            const simplified_data = response.data
            const params = [ 0, response.data, showDefinitions]
            navigate('/output', 
                {state: params}
            )
            
            })
            .catch(error => {
            console.log(error);
            });
        }
        else if(summIsClicked){
            const dataToSummarize = {
                "text": text,
                "summarizeOrSimplify": 1,   // 2 for translate, 1 for summary, 0 for simplify
                "extractiveOrAbstractive":extractiveOrAbstractive,
                "compressedLength":sliderValue / 10
            }
            axios.post('http://localhost:8000/main',  dataToSummarize)
            .then(response => {
            const params = [ 1, response.data]
            navigate('/output', 
                {state: params}
            )
            })
            .catch(error => {
            console.log(error);
            });
        }
        else if(tranIsClicked){
            const dataToTranslate = {
                "text": text,
                "summarizeOrSimplify": 2,   // 2 for translate, 1 for summary, 0 for simplify
            }
            const param1 = 2
            
            axios.post('http://localhost:8000/main',  dataToTranslate)
            .then(response => {
            const params = [ 2, response.data]
            navigate('/output', 
                {state: params}
            )
            })
            .catch(error => {
            console.log(error);
            });
        }
    }

    const handleSimButtonClick = () =>{
        setTranIsClicked(false);
        setSummIsClicked(false);
        setSimIsClicked(true);
    }

    const handleSummButtonClick = () =>{
        setSimIsClicked(false);
        setTranIsClicked(false);
        setSummIsClicked(true);
    }

    const handleTranButtonClick = () => {
        setSimIsClicked(false);
        setSummIsClicked(false);
        setTranIsClicked(true);
    }

    const handleTextInput = (e) => {
        setText(e.target.value);
    }

    const handleFileUpload = (e) => {
        setFileUpload(true);
        files = e.target.files;
        setFileName(files[0].name)
        let reader = new FileReader();
        reader.readAsDataURL(files[0]);
        reader.onload = (e) =>{
            setFile(e.target.result)
        }
    }

    const handleDeleteIconClick = () => {
        setFileName('');
        setFileUpload(false);
        setFile('');
        setText('');
    }

    const changeSummarizationMethod = (e) => {
        setExtractiveOrAbstractive(e.target.value);
    };

    const changeShowDefinitions = (e) => {
        setShowDefinitions(e.target.value);
    }

    const getSliderBackgroundSize = () => {
        return { backgroundSize: `${(sliderValue * 100)/10}% 100%`};
    };

    const handleMouseOverSimp = () => {
        setIsHoveringSimp(true);
    };

    const handleMouseOver = () => {
        setIsHovering(true);
    };
    
    const handleMouseOut = () => {
        setIsHovering(false);   
    };

    const handleMouseOutSimp = () => {
        setIsHoveringSimp(false); 
    }

    useEffect(() => {
        if(file){
            fetch(file)
                .then(response => response.text())
                .then(data => setText(data));         
        }
      }, [file]);

  return (
    <div className="div_container">
        <button className="button_go" onClick={handleGoButtonClick}> GO
        <TfiWrite className="icon_write" size={25}/>  
        </button>
        <label for="myfile" className="label_file">
        <GoCloudUpload className="icon_upload" size={25}/> Upload File</label>
        <input type="file" id="myfile" className="input_file" onChange={handleFileUpload}></input>
        
        
        {fileUpload ? 
        <div className="div_textarea">
            <AiTwotoneDelete className="icon_delete" size={40} onClick={handleDeleteIconClick}/>
            <p class="tooltiptext">Delete File</p>
            <p className="p_filenName">
            <AiTwotoneFileAdd className="icon_file"/>
                {fileName}
            </p>
        </div> : 
        <textarea className={text? "void_textarea" : "input_textarea"} placeholder="Write something here..." style={{resize: 'none'}} onChange={handleTextInput}>
        </textarea> 
        }
        <div className="div_buttons">
        <button className={tranIsClicked ? "button_translate_colored" : "button_translate"} onClick={handleTranButtonClick}>
                    Translate
                </button>
            <button className={simIsClicked ? "button_simplify_colored" : "button_simplify"} onClick={handleSimButtonClick} onMouseOver={handleMouseOverSimp}
            onMouseOut={handleMouseOutSimp}>
                    Simplify
                </button>
            
            <button className={summIsClicked ? "button_summarize_colored" : "button_summarize"} onClick={handleSummButtonClick} onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}>
                Summarize
            </button>
        </div>

        {(isHoveringSimp || simIsClicked) && (
        <div className="div_methods_simp">
            <label><input className="input_button1" type="radio" value='1' checked={showDefinitions == 1} onChange={changeShowDefinitions}/>
                    Show definitions
            </label>
            <br />
            <label>
            <input type="radio" className="input_button2" value='0' checked={showDefinitions == 0} onChange={changeShowDefinitions}/>
                    Do not show definitions
            </label>
        </div>)}

        {(isHovering || summIsClicked)&& (
                <div className="div_methods">
                <div>
                    <label><input className="input_button1" type="radio" value='1' checked={extractiveOrAbstractive == 1} onChange={changeSummarizationMethod}/>
                    Extractive Summary
                    </label>
                    <br />
                    <label>
                    <input type="radio" className="input_button2" value='0' checked={extractiveOrAbstractive == 0} onChange={changeSummarizationMethod}/>
                    Abstractive Summary
                    </label>
                </div>

                    <div className="div_compression">
                        <label>
                        Compressed to : {sliderValue/10}
                        <input type='range' className='input_range' max={10} min={1} value={sliderValue}
                        onChange={(e) => setSliderValue(e.target.valueAsNumber)}
                        style = {getSliderBackgroundSize()}/>
                        </label>
                    </div>
                </div>
            )}
    </div>
  );
}

export default Main;
