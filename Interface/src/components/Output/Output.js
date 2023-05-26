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

function Output() {
    const [searchparams] = useSearchParams();
    const location = useLocation();
    const received_data = location.state;
    console.log(received_data);

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
    if (Array.isArray(received_data)) {
      textAndDetails = received_data;
    } else {
      textAndDetails = [received_data];
    }

    const handleDeleteIconClick = () => {
        setFileName('');
        setFileUpload(false);
        setFile('');
    }


  return (
    <div className="div_container">
    {
      textAndDetails.map((item, index) => (
        <p className="p_output" key={index}>{item}</p>
      ))
    }

    </div>
  );
}

export default Output;
