#ifndef DetectorEvent_h
#define DetectorEvent_h

#include <map>
#include <vector>
#include <string>
#include <TObject.h>

class chip_event {
    public:
        // hit map of the chip in the event
        std::vector<std::pair<
            std::uint16_t,std::uint16_t>> hits;
    
        // ALPIDE readout flags
        bool is_busy_violation;
        bool is_flushed_incomplete;
        bool is_strobe_extended;
        bool is_busy_transition;

        // MOSAIC readout flags
        bool end_of_run;
        bool overflow;
        bool timeout;
        bool header_error;
        bool decoder_10b8b_error;
        bool event_oversize_error;

        // chip identificators
        std::uint8_t chip_id;
        std::uint8_t channel;

        ClassDef(chip_event, 2);
};

class tlu_event {
    public:
        // fine trigger timestamp in 
        // 1.5 ns units showing
        // relative time of trigger
        // arrival 
        std::uint8_t fine_ts_0;
        std::uint8_t fine_ts_1;
        std::uint8_t fine_ts_2;
        std::uint8_t fine_ts_3;
        std::uint8_t fine_ts_4;
        std::uint8_t fine_ts_5;
    
        // number of triggers pre-veto (e.g. 
        // counts triggers under DUT-asserted 
        // busy)
        std::uint32_t particles;
        
        // event counnters in each trgger 
        // input (above threshold but not 
        // necessary unmasked)
        std::uint32_t scaler_0;
        std::uint32_t scaler_1;
        std::uint32_t scaler_2;
        std::uint32_t scaler_3;
        std::uint32_t scaler_4;
        std::uint32_t scaler_5;
    
        // shows trigger inputs
        // active in an event
        std::string trg_sign;
    
        // on of the following event types: 
        // 0000 trigger internal; 0001 trigger external
        // 0010 shutter falling;  0011 shutter rising
        // 0100 edge falling;     0101 edge rising
        // 0111 spill on;         0110 spill off
        std::uint8_t type;
    
        // event trigger id
        std::uint16_t trg_n;
    
        // event timestamp in ns
        std::uint64_t event_begin;
        std::uint64_t event_end;
    
        ClassDef(tlu_event, 1);
};

class stave_event {
    public:
        // chip event storage
        std::vector<chip_event> ch_ev_buffer;
    
        // stave identificator
        std::uint8_t stave_id;
    
        ClassDef(stave_event, 1);
};

class detector_event {
    public:
        // stave event storage
        std::vector<stave_event> st_ev_buffer;
        
        // event MOSAIC trigger ID
        std::uint32_t trg_n;
    
        ClassDef(detector_event, 2);
};

class detector_event_tlu {
    public:
        // stave event storage
        std::vector<stave_event> st_ev_buffer;
    
        // to sync with the tlu
        tlu_event tl_ev;
    
        ClassDef(detector_event_tlu, 2);
};

#endif
