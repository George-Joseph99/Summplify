import React, { useState, useEffect } from 'react' ;
import axios from "axios";
import { GoCloudUpload } from "react-icons/go";
import { AiTwotoneFileAdd } from "react-icons/ai";
import { AiTwotoneDelete } from "react-icons/ai";
import './Main.css';
import { List, Card, Button } from 'antd';
import { createSearchParams, useNavigate } from 'react-router-dom';

function Main() {

    const [fileUpload, setFileUpload] = useState(false);
    const [fileName, setFileName] = useState('');
    const [file, setFile] = useState('');
    const [text, setText] = useState('');
    const [extractiveOrAbstractive, setExtractiveOrAbstractive] = useState('1');
    const [sliderValue, setSliderValue] = useState(3);
    var files = [];
    const navigate = useNavigate();

    const handleSimButtonClick = () =>{
        console.log("clickeddd");
        const dataToSimplify = {
            "text": text,
            "summarizeOrSimplify": 0   // 1 for summary, 0 for simplify
        }
        axios.post('http://localhost:5000/main',  dataToSimplify)
        .then(response => {
        console.log(response.data);
        navigate({
            pathname : "/output",
            search: createSearchParams({
                text : response.data
            }).toString()
            })
        })
        .catch(error => {
        console.log(error);
        });
    }

    const handleSummButtonClick = () =>{
        console.log("clickeddd");
        const dataToSummarize = {
            "text": text,
            "summarizeOrSimplify": 1,   // 1 for summary, 0 for simplify
            "extractiveOrAbstractive":extractiveOrAbstractive,
            "compressedLength":sliderValue / 10
        }
        axios.post('http://localhost:5000/main',  dataToSummarize)
        .then(response => {
        console.log(response.data);
        navigate({
            pathname : "/output",
            search: createSearchParams({
                text : response.data
            }).toString()
            })
        })
        .catch(error => {
        console.log(error);
        });
    }

    const handleTextInput = (e) => {
        console.log(e.target.value);
        setText(e.target.value);
    }

    const handleFileUpload = (e) => {
        setFileUpload(true);
        files = e.target.files;
        console.log(files);
        setFileName(files[0].name)
        let reader = new FileReader();
        reader.readAsDataURL(files[0]);
        reader.onload = (e) =>{
            setFile(e.target.result)
            console.log("img data", e.target.result);
        }
    }

    const handleDeleteIconClick = () => {
        setFileName('');
        setFileUpload(false);
        setFile('');
    }

    const changeSummarizationMethod = (e) => {
        setExtractiveOrAbstractive(e.target.value);
    };

    const getSliderBackgroundSize = () => {
        return { backgroundSize: `${(sliderValue * 100)/10}% 100%`};
    };

  return (
    <div className="div_container">
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
            <label for="myfile" className="label_file">
            <GoCloudUpload className="icon_upload" size={25}/> Upload File</label>
            <input type="file" id="myfile" className="input_file" onChange={handleFileUpload}></input>
            <button className="button_simplify" onClick={handleSimButtonClick}>
                Simplify
            </button>
            <button className="button_summarize" onClick={handleSummButtonClick}>
                Summarize
            </button>

            <div>
            <label><input type="radio" value='1' checked={extractiveOrAbstractive == 1} onChange={changeSummarizationMethod}/>
            Extractive Summary
            </label>
            <br />
            <label>
            <input type="radio" value='0' checked={extractiveOrAbstractive == 0} onChange={changeSummarizationMethod}/>
            Abstractive Summary
            </label>
            </div>

            <div >
                <label>
                Compressed to : {sliderValue/10}
                <input type='range' className='input_range' max={10} min={1} value={sliderValue}
                onChange={(e) => setSliderValue(e.target.valueAsNumber)}
                style = {getSliderBackgroundSize()}/>
                </label>
            </div>
        </div>
    </div>
  );
}

export default Main;
