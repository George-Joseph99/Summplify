import React, { useState, useEffect } from 'react' ;
import axios from "axios";
import { GoCloudUpload } from "react-icons/go";
import { AiTwotoneFileAdd } from "react-icons/ai";
import { AiTwotoneDelete } from "react-icons/ai";
import './Main.css';
import { List, Card, Button } from 'antd';
function Main() {

    const [fileUpload, setFileUpload] = useState(false);
    const [fileName, setFileName] = useState('');
    const [file, setFile] = useState('');
    const [text, setText] = useState('');
    var files = [];

    const handleSimButtonClick = () =>{
        console.log("clickeddd");
        console.log(fileUpload);
    }

    const handleSummButtonClick = () =>{
        console.log("clickeddd");
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
        </div>
    </div>
  );
}

export default Main;
