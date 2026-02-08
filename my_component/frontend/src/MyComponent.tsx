import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

/**
 * Global declaration for the JSME applet, which is loaded externally.
 */
declare global {
  interface Window {
    JSApplet: any;
  }
}

/**
 * Interface for the JSON structure used for communication between Python and the component.
 * This structure holds the SMILES string and a unique ID for request tracking.
 */
interface SmilesRequest {
    smiles: string;
    id: string; // Unique identifier for the request
}

/**
 * Component state management.
 */
interface State {
  jsmeLoaded: boolean;
  status: string; // Internal processing status
  currentRequestId: string | null; // ID of the last processed/in-progress request
}

class JSMEComponent extends StreamlitComponentBase<State> {
  private jsmeApplet: any = null;
  // Tracker for the last JSON input received from Streamlit
  private currentInputJson: string = ''; 

  constructor(props: any) {
    super(props);
    // Initialize component state
    this.state = { jsmeLoaded: false, status: 'Initial', currentRequestId: null }; 
  }

  componentDidMount(): void {
    // Start the JSME loading sequence when the component mounts
    this.loadJSME();
  }

  componentDidUpdate(): void {
    // Python input is received via props.args.smiles_json
    const nextInputJson = this.props.args?.smiles_json || null;
    let nextRequest: SmilesRequest | null = null;
    
    // --- 1. Check and Parse Input ---
    // Only attempt parsing if the input JSON string is new and not null
    if (nextInputJson && nextInputJson !== this.currentInputJson) {
        try {
            nextRequest = JSON.parse(nextInputJson);
        } catch (e) {
            console.error("Failed to parse input JSON:", e);
            this.currentInputJson = nextInputJson; // Block processing of this malformed input
            return; 
        }
    }
    
    // --- 2. Processing Trigger Logic ---
    if (this.state.jsmeLoaded && nextRequest && nextRequest.smiles.trim() !== '') {
        // Trigger processing only if the SMILES is valid AND the request ID is new
        if (nextRequest.id !== this.state.currentRequestId) {
            this.currentInputJson = nextInputJson; // Update full JSON tracker
            this.setState({ 
                status: 'Processing...',
                currentRequestId: nextRequest.id 
            }); 
            // Initiate the safe processing immediately
            this.safeProcessSmiles(nextRequest.smiles, nextRequest.id); 
        }
    }
    
    // --- 3. Cleanup/Reset Logic ---
    // If Python sends null, it signals a reset or a step completion. 
    // We send an empty JSON back to force Python to re-evaluate the state.
    if (nextInputJson === null) {
        Streamlit.setComponentValue(JSON.stringify({})); 
        this.currentInputJson = ''; // Reset input tracker
        this.setState({ status: 'Idle', currentRequestId: null }); 
    }
  }

  /**
   * Loads the external JSME script dynamically.
   */
  loadJSME = (): void => {
    if (window.JSApplet) {
      this.initJSME();
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://jsme-editor.github.io/dist/jsme/jsme.nocache.js';
    script.async = true;
    script.onload = () => this.waitForJSME();
    document.head.appendChild(script);
  }

  /**
   * Waits for the JSApplet object to be globally available.
   */
  waitForJSME = (): void => {
    if (window.JSApplet) {
      this.initJSME();
    } else {
      setTimeout(() => this.waitForJSME(), 100);
    }
  }

  /**
   * Initializes the JSME applet in a hidden container for non-visual processing.
   */
  initJSME = (): void => {
    // Crucial: Create the JSME in a hidden DIV so it doesn't interfere with the UI
    const hiddenDiv = document.createElement('div');
    hiddenDiv.style.position = 'absolute';
    hiddenDiv.style.left = '-9999px';
    hiddenDiv.style.width = '200px'; 
    hiddenDiv.style.height = '200px';
    hiddenDiv.id = 'jsme-processor';
    document.body.appendChild(hiddenDiv);

    // Instantiate the JSME applet
    this.jsmeApplet = new window.JSApplet.JSME("jsme-processor", "200px", "200px");
    
    this.setState({ jsmeLoaded: true, status: 'Ready' }); 
    
    // Set component height to minimal value since it's an invisible processor
    Streamlit.setFrameHeight(10);
  }
  
  /**
   * Checks if the JSME applet object is initialized and ready to receive commands.
   */
  isJSMEAppletReady = (): boolean => {
    return this.jsmeApplet && typeof this.jsmeApplet.smiles === 'function';
  }

  /**
   * Attempts to process the SMILES input using JSME, with retries if the applet is not ready
   * and a robust timeout to wait for the internal JSME processing to complete.
   */
  safeProcessSmiles = (smiles: string, requestId: string, attempt: number = 0): void => {
    const MAX_ATTEMPTS = 20;
    
    // 1. Check for max attempts
    if (attempt >= MAX_ATTEMPTS) {
        console.error("JSME Processing failed: Max retry attempts reached.");
        // Return a specific error value with the ID for tracking
        const errorOutput = JSON.stringify({ smiles: `JSME_ERROR:TIMEOUT`, id: requestId });
        Streamlit.setComponentValue(errorOutput); 
        this.setState({ status: 'Failed (Timeout)' });
        return;
    }

    // 2. Process SMILES if ready
    if (this.isJSMEAppletReady()) {
      try {
        // Load the molecular input into the hidden JSME instance
        this.jsmeApplet.readGenericMolecularInput(smiles);
        
        // Wait for JSME's internal processing cycle to complete (robustness delay)
        setTimeout(() => {
             const processedSmiles = this.jsmeApplet.smiles() || "JSME_ERROR:EMPTY_OUTPUT"; 
        
             // Return the structured result with the request ID
             const finalOutput = JSON.stringify({ 
                 smiles: processedSmiles,
                 id: requestId
             });
             Streamlit.setComponentValue(finalOutput);
             this.setState({ status: 'Processed' });
        }, 100); // Increased from 50ms to 100ms for better robustness
        
      } catch (error) {
        console.error("Error processing SMILES in JSME:", error);
        const errorOutput = JSON.stringify({ smiles: `JSME_ERROR:EXCEPTION`, id: requestId });
        Streamlit.setComponentValue(errorOutput);
        this.setState({ status: 'Failed (Exception)' });
      }
    } else {
      // 3. Retry if not ready
      console.log(`JSME not ready, retrying... Attempt: ${attempt + 1}`);
      setTimeout(() => this.safeProcessSmiles(smiles, requestId, attempt + 1), 100);
    }
  }

  public render = (): ReactNode => {
    const { jsmeLoaded, status } = this.state;

    // Render a minimal indicator of the processor's status for developer visibility.
    // This element is hidden from the user interface.
    return (
      <div style={{
        padding: '5px 10px',
        fontSize: '10px',
        color: '#666',
        backgroundColor: jsmeLoaded ? '#e0ffe0' : '#ffeee0',
        borderRadius: '2px',
        display: 'inline-block',
        minWidth: '200px'
      }}>
        {jsmeLoaded ? `✓ JSME Processor Ready. Status: ${status}` : '⏳ Loading JSME Processor...'}
      </div>
    );
  }
}

export default withStreamlitConnection(JSMEComponent)